"""Permission-aware retrieval test on the real corpus.

A local user may search, read and cite only a hash-chosen half of the
documents. Every question that HEAD can answer is replayed as that user.
Expected: zero citations or presented passages from the forbidden half, and a
much lower answer rate when the gold document is in the forbidden half.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from typing import Any, Mapping

from .common import append_jsonl, read_jsonl, write_json
from .runner import (Variant, answer_record, default_engine_factory, index_snapshot, make_config,
                     snapshot_mapping, snapshot_tree, vector_cache_for, workspace_for)

USER, GROUP = "eval-acl-user", "eval-acl-half"
PERMISSIONS = ("document.search", "document.read", "document.cite")


def in_allowed_half(repo_path: str) -> bool:
    return hashlib.sha256(repo_path.encode("utf-8")).digest()[0] % 2 == 0


def _repo_path(document, tree: Path, mapping: Mapping[str, str]) -> str | None:
    try:
        indexed = Path(document.source_path).resolve().relative_to(tree.resolve()).as_posix()
    except ValueError:
        return None
    return mapping.get(indexed, indexed)


def _setup_access(engine, tree: Path, mapping: Mapping[str, str]) -> set[str]:
    """Create the user/group/grants once; return the allowed repository paths."""

    access = engine.access
    documents = [(d, _repo_path(d, tree, mapping)) for d in engine.store.documents()]
    allowed = {path for _, path in documents if path is not None and in_allowed_half(path)}
    if not access.principal_snapshot(USER).known:
        access.create_user(USER, "Evaluation ACL user")
        access.assign_system_role(USER, "viewer")
        access.grant(USER, "*", "*", "question.ask")
        access.create_group(GROUP, "Evaluation allowed half")
        access.add_group_member(GROUP, USER)
        for document, path in documents:
            if path in allowed:
                for permission in PERMISSIONS:
                    access.grant(GROUP, "document", document.logical_document_id, permission,
                                 principal_type="group")
    return allowed


def run_acl(data_dir: Path, project_name: str, variant: Variant, questions: list[Mapping[str, Any]], *,
            limit: int | None = None, engine_factory=None, log=print) -> dict[str, Any]:
    eligible = [q for q in questions if q.get("answerable_at_head")]
    if not eligible:
        raise ValueError("No questions are answerable at HEAD; run `snapshot` first")
    commit = eligible[0]["head_commit"]
    if limit is not None:
        eligible = eligible[:limit]
    index_snapshot(data_dir, project_name, commit, variant, purpose="acl")
    workspace = workspace_for(data_dir, project_name, commit, purpose="acl")
    cache = vector_cache_for(data_dir, project_name)
    tree = snapshot_tree(data_dir, project_name, commit)
    mapping = snapshot_mapping(data_dir, project_name, commit)
    config = make_config(variant, workspace, vector_cache=cache)
    engine = (engine_factory or default_engine_factory)(variant, config, vector_cache=cache)
    output = data_dir / project_name / "acl" / f"{variant.name}.jsonl"
    done = {r["qid"] for r in read_jsonl(output)}
    try:
        allowed = _setup_access(engine, tree, mapping)
        for question in eligible:
            if question["qid"] in done:
                continue
            gold_paths = {g["path"] for g in question["gold"] if g.get("status_at_head") == "present"}
            row: dict[str, Any] = {"qid": question["qid"], "variant": variant.name,
                                   "gold_allowed": bool(gold_paths & allowed)}
            try:
                answer = engine.ask(question["question"], session=None, mode="auto", principal=USER)
                record = answer_record(answer, engine=engine, tree=tree, mapping=mapping)
                row.update({"answered": record["answered"], "status": record["status"],
                            "leaked_citations": sorted({c["path"] for c in record["citations"]} - allowed),
                            "leaked_presented": sorted(set(record["presented_docs"]) - allowed),
                            "cited_paths": sorted({c["path"] for c in record["citations"]})})
            except PermissionError as exc:
                row.update({"answered": False, "status": "permission_error", "error": str(exc)[:300],
                            "leaked_citations": [], "leaked_presented": []})
            append_jsonl(output, row)
            log(f"[acl] {question['qid']}: gold_allowed={row['gold_allowed']} answered={row['answered']} "
                f"leaks={len(row['leaked_citations']) + len(row['leaked_presented'])}", file=sys.stderr)
    finally:
        engine.close()
    return summarize_acl(data_dir, project_name, variant.name)


def summarize_acl(data_dir: Path, project_name: str, variant_name: str) -> dict[str, Any]:
    rows = read_jsonl(data_dir / project_name / "acl" / f"{variant_name}.jsonl")
    allowed = [r for r in rows if r["gold_allowed"]]
    forbidden = [r for r in rows if not r["gold_allowed"]]
    summary = {"variant": variant_name, "questions": len(rows),
               "answers_with_leaked_citations": sum(1 for r in rows if r["leaked_citations"]),
               "answers_with_leaked_presented_passages": sum(1 for r in rows if r["leaked_presented"]),
               "answer_rate_gold_allowed": (sum(r["answered"] for r in allowed) / len(allowed)) if allowed else None,
               "answer_rate_gold_forbidden": (sum(r["answered"] for r in forbidden) / len(forbidden)) if forbidden else None,
               "n_gold_allowed": len(allowed), "n_gold_forbidden": len(forbidden)}
    write_json(data_dir / project_name / "acl" / f"{variant_name}.summary.json", summary)
    return summary
