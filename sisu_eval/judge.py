"""LLM judge for answer correctness (vs the maintainer's answer) and for
whether the answer is supported by the passages it cites.

Default judge: gpt-oss:20b through the same local Ollama, a different model
family from the Gemma 4 generator to reduce self-preference. The judge is
checked against two human raters with ``calibrate`` before its numbers are
reported.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping

from .common import REPO_ROOT, append_jsonl, read_jsonl
from .runner import results_path, safe_name

DEFAULT_JUDGE = "gpt-oss:20b"
JUDGE_VERSION = "sisu-eval-judge-v1"

CORRECTNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["correct", "partially_correct", "incorrect"]},
        "rationale": {"type": "string"},
    },
    "required": ["verdict", "rationale"],
}
SUPPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["fully_supported", "partially_supported", "unsupported"]},
        "unsupported_statements": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
    "required": ["verdict", "unsupported_statements", "rationale"],
}

CORRECTNESS_SYSTEM = (
    "You grade answers to questions that newcomers asked about a software project. A project maintainer's "
    "reply is the reference. Grade the candidate answer against the reference only:\n"
    "- correct: gives the same resolution as the reference (same key facts, steps or recommendation) and "
    "contradicts nothing in it. Extra correct detail is fine.\n"
    "- partially_correct: gets part of the resolution but misses a key element, or adds a minor error.\n"
    "- incorrect: wrong, contradicts the reference, or does not address the question.\n"
    "Wording and length do not matter. Reply as JSON with a short rationale (under 60 words).")
SUPPORT_SYSTEM = (
    "You check whether an answer is supported by the source passages it cites. Judge only against the "
    "passages, not your own knowledge.\n"
    "- fully_supported: every factual statement is stated in or directly follows from the passages.\n"
    "- partially_supported: at least one statement is not backed by the passages.\n"
    "- unsupported: the main statements are not backed by the passages.\n"
    "List the unsupported statements. Reply as JSON with a short rationale (under 60 words).")


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + " [...]"


def correctness_messages(question: Mapping[str, Any], answer_text: str) -> list[dict[str, str]]:
    payload = {"question": _clip(question["question"], 3000),
               "reference_answer_by_maintainer": _clip(question["answer"], 3000),
               "candidate_answer": _clip(answer_text, 4000)}
    return [{"role": "system", "content": CORRECTNESS_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def support_messages(answer_text: str, citations: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    passages, used = [], 0
    for index, citation in enumerate(citations, 1):
        quote = _clip(citation.get("quote", ""), 2000)
        if used + len(quote) > 9000:
            break
        used += len(quote)
        passages.append({"id": f"S{index}", "source": citation.get("path", ""), "text": quote})
    payload = {"answer": _clip(answer_text, 4000), "cited_passages": passages}
    return [{"role": "system", "content": SUPPORT_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def make_client(judge_model: str):
    from sisu_reader.config import Config
    from sisu_reader.ollama import OllamaClient

    workspace = Path(tempfile.gettempdir()) / "sisu_eval_judge"
    config = Config(REPO_ROOT, workspace, model=judge_model, reasoning="off", request_timeout_s=180.0,
                    total_deadline_s=200.0, synthesis_output_tokens=3000, review_output_tokens=3000)
    return OllamaClient(config, model=judge_model)


def _ask(client, messages, schema) -> dict[str, Any]:
    result = client.chat(messages=messages, format=schema, num_predict=2048, temperature=0.0, seed=7)
    return json.loads(result.content)


def judge_row(client, question: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    if not row.get("answered"):
        return {"correctness": {"verdict": "no_answer", "rationale": "System returned no answer."},
                "support": None, "auto": True}
    out: dict[str, Any] = {"auto": False}
    try:
        out["correctness"] = _ask(client, correctness_messages(question, row["text"]), CORRECTNESS_SCHEMA)
    except Exception as exc:
        out["correctness"] = {"verdict": "judge_error", "rationale": f"{type(exc).__name__}: {exc}"[:300]}
    if row.get("citations"):
        try:
            out["support"] = _ask(client, support_messages(row["text"], row["citations"]), SUPPORT_SCHEMA)
        except Exception as exc:
            out["support"] = {"verdict": "judge_error", "unsupported_statements": [],
                              "rationale": f"{type(exc).__name__}: {exc}"[:300]}
    else:
        out["support"] = {"verdict": "uncited", "unsupported_statements": [], "rationale": "No citations."}
    return out


def judgments_path(data_dir: Path, project_name: str, judge_model: str) -> Path:
    return data_dir / project_name / "judgments" / f"{safe_name(judge_model)}.jsonl"


def judge_variants(data_dir: Path, project_name: str, variant_names: Iterable[str],
                   questions: Mapping[str, Mapping[str, Any]], *, judge_model: str = DEFAULT_JUDGE,
                   client=None, log=print) -> dict[str, int]:
    path = judgments_path(data_dir, project_name, judge_model)
    done = {(r["variant"], r["qid"], r["corpus"]) for r in read_jsonl(path)
            if r["correctness"]["verdict"] != "judge_error"}
    client = client or make_client(judge_model)
    counts = {"judged": 0, "skipped": 0}
    for variant in variant_names:
        for row in read_jsonl(results_path(data_dir, project_name, variant)):
            key = (row["variant"], row["qid"], row["corpus"])
            if key in done or row["qid"] not in questions or row.get("run_error"):
                counts["skipped"] += 1
                continue
            verdicts = judge_row(client, questions[row["qid"]], row)
            append_jsonl(path, {"variant": row["variant"], "qid": row["qid"], "corpus": row["corpus"],
                                "judge_model": judge_model, "judge_version": JUDGE_VERSION, **verdicts})
            done.add(key)
            counts["judged"] += 1
            log(f"[judge {judge_model}] {variant} {row['qid']}: {verdicts['correctness']['verdict']}",
                file=sys.stderr)
    return counts


def load_judgments(data_dir: Path, project_name: str, judge_model: str) -> dict[tuple[str, str, str], dict]:
    latest = {}
    for row in read_jsonl(judgments_path(data_dir, project_name, judge_model)):
        latest[(row["variant"], row["qid"], row["corpus"])] = row
    return latest
