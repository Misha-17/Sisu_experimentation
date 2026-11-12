"""Index dated snapshots and run every variant over the mined questions."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
import time
import traceback
from typing import Any, Callable, Iterable, Mapping

from .common import PACKAGE_DIR, REPO_ROOT, append_jsonl, environment, read_json, read_jsonl

VARIANTS_FILE = PACKAGE_DIR / "variants.json"
REFUSAL_TEXT = "I could not establish an answer from the inspected source passages."
ANSWERED_STATUSES = {"answer", "partial"}


# Variants -----------------------------------------------------------------------

@dataclass(frozen=True)
class Variant:
    name: str
    engine: str
    env: Mapping[str, str]
    description: str


def load_variants(path: Path | None = None) -> tuple[dict[str, Variant], str]:
    data = read_json(path or VARIANTS_FILE)
    raw = data["variants"]
    resolved: dict[str, Variant] = {}

    def resolve(name: str, chain: tuple[str, ...] = ()) -> Variant:
        if name in resolved:
            return resolved[name]
        if name in chain:
            raise ValueError(f"Variant inheritance loop: {' -> '.join(chain + (name,))}")
        item = raw[name]
        parent = resolve(item["extends"], chain + (name,)) if item.get("extends") else None
        env = {**data.get("common", {}), **(dict(parent.env) if parent else {}), **item.get("env", {})}
        variant = Variant(name, item.get("engine", parent.engine if parent else "sisu"),
                          {k: str(v) for k, v in env.items()}, item.get("description", ""))
        resolved[name] = variant
        return variant

    for name in raw:
        resolve(name)
    reference = data.get("reference", "normal")
    if reference not in resolved:
        raise ValueError(f"Reference variant {reference!r} is not defined")
    return resolved, reference


def select_variants(all_variants: Mapping[str, Variant], names: str | None) -> list[Variant]:
    if not names:
        return list(all_variants.values())
    chosen = []
    for name in (x.strip() for x in names.split(",") if x.strip()):
        if name not in all_variants:
            raise ValueError(f"Unknown variant {name!r}; defined: {', '.join(all_variants)}")
        chosen.append(all_variants[name])
    return chosen


# Engines ----------------------------------------------------------------------------

def make_config(variant: Variant, workspace: Path, *, vector_cache: Path, principal: str | None = None):
    from sisu_reader.config import Config

    overrides = {**variant.env, "SISU_READER_WORKSPACE": str(workspace),
                 "SISU_READER_VECTOR_CACHE_DIR": str(vector_cache)}
    if principal:
        overrides["SISU_READER_USER_ID"] = principal
    with environment(overrides, clear_prefix="SISU_READER_"):
        return Config.load(REPO_ROOT)


def default_engine_factory(variant: Variant, config, *, vector_cache: Path):
    with environment({"SISU_READER_VECTOR_CACHE_DIR": str(vector_cache)}):
        if variant.engine == "plain_rag":
            from .baseline import PlainRag
            engine = PlainRag(config)
        else:
            from sisu_reader.engine import SisuReader
            engine = SisuReader(config)
        engine.load()
    return engine


EngineFactory = Callable[..., Any]


# Paths -------------------------------------------------------------------------------

def project_dir(data_dir: Path, project_name: str) -> Path:
    return data_dir / project_name


def workspace_for(data_dir: Path, project_name: str, commit: str, *, purpose: str = "eval") -> Path:
    return project_dir(data_dir, project_name) / "workspaces" / purpose / commit[:16]


def vector_cache_for(data_dir: Path, project_name: str) -> Path:
    return project_dir(data_dir, project_name) / "vector_cache"


def snapshot_tree(data_dir: Path, project_name: str, commit: str) -> Path:
    return project_dir(data_dir, project_name) / "snapshots" / commit / "tree"


def snapshot_mapping(data_dir: Path, project_name: str, commit: str) -> dict[str, str]:
    manifest = project_dir(data_dir, project_name) / "snapshots" / commit / "manifest.json"
    return read_json(manifest)["indexed_to_repo_path"]


# Indexing --------------------------------------------------------------------------

def index_snapshot(data_dir: Path, project_name: str, commit: str, variant: Variant, *,
                   purpose: str = "eval", engine_class=None) -> dict[str, Any]:
    workspace = workspace_for(data_dir, project_name, commit, purpose=purpose)
    marker = workspace / "indexed.json"
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf-8"))
    tree = snapshot_tree(data_dir, project_name, commit)
    cache = vector_cache_for(data_dir, project_name)
    config = make_config(variant, workspace, vector_cache=cache)
    with environment({"SISU_READER_VECTOR_CACHE_DIR": str(cache)}):
        if engine_class is None:
            from sisu_reader.engine import SisuReader as engine_class
        engine = engine_class(config)
        try:
            report = engine.rebuild([tree])
            stats = engine.retrieval.ensure()
        finally:
            engine.close()
    summary = {"commit": commit, "documents": report.documents, "blocks": report.blocks,
               "warnings": list(report.warnings)[:50], "passages": stats.get("passages"),
               "dense_available": stats.get("dense_available"),
               "dense_fallback_reason": stats.get("dense_fallback_reason")}
    marker.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not stats.get("dense_available"):
        print(f"warning: snapshot {commit[:12]} indexed WITHOUT the E5 encoder ({stats.get('dense_fallback_reason')}); "
              "run scripts/download_encoder.py first or dense/hybrid variants are not meaningful", file=sys.stderr)
    return summary


# Answer records ------------------------------------------------------------------

def _relative(path: str, tree: Path, mapping: Mapping[str, str]) -> str:
    try:
        indexed = Path(path).resolve().relative_to(tree.resolve()).as_posix()
    except ValueError:
        return Path(path).name
    return mapping.get(indexed, indexed)


def _call_rows(trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for call in trace.get("model_calls", ()):
        metrics = call.get("metrics", {}) or {}
        rows.append({"role": call.get("role"), "status": call.get("status"), "elapsed_s": call.get("elapsed_s"),
                     "eval_count": metrics.get("eval_count"), "prompt_eval_count": metrics.get("prompt_eval_count"),
                     "thinking_chars": metrics.get("thinking_chars"), "think_sent": metrics.get("think_sent"),
                     "num_predict": metrics.get("num_predict"), "error": (call.get("error") or "")[:300]})
    return rows


def answer_record(answer: Any, *, engine: Any, tree: Path, mapping: Mapping[str, str]) -> dict[str, Any]:
    """Normalize a SISU ``Answer`` (or the baseline's equivalent) for scoring."""

    debug = dict(getattr(answer, "debug", {}) or {})
    trace: dict[str, Any] = {}
    trace_path = getattr(answer, "trace_path", "") or ""
    if trace_path and Path(trace_path).exists():
        trace = json.loads(Path(trace_path).read_text(encoding="utf-8"))
    elif "trace" in debug:
        trace = debug["trace"]
    documents = {d.document_revision_id: d.source_path for d in engine.store.documents()}
    citations = []
    for source in getattr(answer, "sources", ()) or ():
        get = source.get if isinstance(source, Mapping) else (lambda key, s=source: getattr(s, key, ""))
        citations.append({"path": _relative(get("source_path"), tree, mapping), "title": get("title"),
                          "locator": get("locator"), "quote": get("quote"), "claim_id": get("card_id")})
    presented_docs = sorted({_relative(documents[d], tree, mapping) for d in debug.get("read_document_ids", ())
                             if d in documents})
    sections = []
    for card in trace.get("evidence_cards", ()) or debug.get("evidence_cards", ()):
        path = documents.get(card.get("document_revision_id"), card.get("source_path", ""))
        sections.append({"path": _relative(path, tree, mapping) if path else "",
                         "section_path": card.get("section_path", ""), "locator": card.get("locator", ""),
                         "headers": card.get("headers", [])})
    timings = dict(getattr(answer, "timings", {}) or {})
    text = getattr(answer, "text", "") or ""
    status = getattr(answer, "status", "error")
    return {"status": status, "text": text,
            "answered": status in ANSWERED_STATUSES and text.strip() != REFUSAL_TEXT,
            "citations": citations, "presented_docs": presented_docs, "presented_sections": sections,
            "warnings": list(getattr(answer, "warnings", ()) or ()), "errors": debug.get("errors", []),
            "facet_gaps": debug.get("facet_gaps", []), "facets": len((debug.get("question_plan") or {}).get("facets", [])),
            "ablation": debug.get("ablation", {}), "timings": timings,
            "total_s": timings.get("total_s", timings.get("answer_ready_s")),
            "model_calls": _call_rows(trace) or debug.get("model_call_rows", [])}


# Running -------------------------------------------------------------------------------

def results_path(data_dir: Path, project_name: str, variant_name: str) -> Path:
    return project_dir(data_dir, project_name) / "runs" / f"{variant_name}.jsonl"


def _done(path: Path, *, retry_errors: bool) -> set[tuple[str, str]]:
    done = set()
    for row in read_jsonl(path):
        if retry_errors and row.get("run_error"):
            continue
        done.add((row["qid"], row["corpus"]))
    return done


def plan_jobs(questions: Iterable[Mapping[str, Any]], *, head_check: bool) -> list[tuple[str, Mapping[str, Any]]]:
    """(corpus, question) pairs. Unanswerable-at-snapshot questions are also
    asked against HEAD when ``head_check`` is set and HEAD has the evidence."""

    jobs = []
    for question in questions:
        jobs.append(("snapshot", question))
        if head_check and not question["answerable_at_snapshot"] and question.get("answerable_at_head"):
            jobs.append(("head", question))
    return jobs


def run_variant(data_dir: Path, project_name: str, variant: Variant, questions: list[Mapping[str, Any]], *,
                head_check: bool = False, retry_errors: bool = False, limit: int | None = None,
                engine_factory: EngineFactory | None = None, log=print) -> dict[str, Any]:
    output = results_path(data_dir, project_name, variant.name)
    done = _done(output, retry_errors=retry_errors)
    jobs = [job for job in plan_jobs(questions, head_check=head_check)
            if (job[1]["qid"], job[0]) not in done]
    if limit is not None:
        jobs = jobs[:limit]
    by_commit: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
    for corpus, question in jobs:
        commit = question["snapshot_commit"] if corpus == "snapshot" else question["head_commit"]
        by_commit.setdefault(commit, []).append((corpus, question))
    cache = vector_cache_for(data_dir, project_name)
    factory = engine_factory or default_engine_factory
    completed = errors = 0
    for commit, items in by_commit.items():
        index_snapshot(data_dir, project_name, commit, variant)
        workspace = workspace_for(data_dir, project_name, commit)
        config = make_config(variant, workspace, vector_cache=cache)
        tree = snapshot_tree(data_dir, project_name, commit)
        mapping = snapshot_mapping(data_dir, project_name, commit)
        engine = factory(variant, config, vector_cache=cache)
        try:
            for corpus, question in items:
                started = time.perf_counter()
                row = {"variant": variant.name, "qid": question["qid"], "corpus": corpus, "commit": commit,
                       "project": project_name, "started_at": time.time()}
                try:
                    answer = engine.ask(question["question"], session=None, mode="auto")
                    row.update(answer_record(answer, engine=engine, tree=tree, mapping=mapping))
                    completed += 1
                except Exception as exc:  # recorded; the run continues
                    row.update({"run_error": f"{type(exc).__name__}: {exc}"[:500],
                                "traceback": traceback.format_exc()[-2000:], "answered": False,
                                "status": "run_error"})
                    errors += 1
                row["wall_s"] = time.perf_counter() - started
                append_jsonl(output, row)
                log(f"[{variant.name}] {question['qid']} ({corpus}): {row.get('status')} in {row['wall_s']:.1f}s",
                    file=sys.stderr)
        finally:
            close = getattr(engine, "close", None)
            if close:
                close()
    return {"variant": variant.name, "jobs": len(jobs), "completed": completed, "errors": errors,
            "output": str(output)}


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
