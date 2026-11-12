"""Aggregate scored runs into report.md, report.json and metrics.csv."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

from .common import read_json, read_jsonl, write_json
from .judge import load_judgments
from .labels import load_labels
from .metrics import METRICS, aggregate, compare, fmt_mean_ci, score_answer
from .runner import Variant, results_path
from .snapshot import load_questions

HEADLINE = ("correct_lenient", "correct_strict", "evidence_doc_recall", "evidence_section_recall",
            "citation_gold_hit", "fully_supported", "false_refusal", "correct_refusal_unanswerable", "within_95s")
COST = ("model_calls", "output_tokens", "thinking_chars", "call_errors")


def _latest(rows):
    latest = {}
    for row in rows:
        latest[(row["qid"], row["corpus"])] = row
    return latest


def build_report(data_dir: Path, project_name: str, variants: list[Variant], reference: str, judge_model: str,
                 project=None) -> dict[str, Any]:
    from .projects import load_project

    project = project or load_project(project_name)
    labels = load_labels(data_dir, project_name)
    questions = {q["qid"]: q for q in load_questions(project, data_dir, labels=labels)}
    judgments = load_judgments(data_dir, project_name, judge_model)
    scored: dict[str, dict[str, dict[tuple[str, str], Mapping[str, Any]]]] = {}
    run_errors = {}
    for variant in variants:
        rows = _latest(read_jsonl(results_path(data_dir, project_name, variant.name)))
        run_errors[variant.name] = sum(1 for r in rows.values() if r.get("run_error"))
        for (qid, corpus), row in rows.items():
            if qid not in questions:
                continue
            scored.setdefault(variant.name, {}).setdefault(corpus, {})[(qid, corpus)] = score_answer(
                questions[qid], row, judgments.get((variant.name, qid, corpus)))
    report: dict[str, Any] = {"project": project_name, "reference": reference, "judge_model": judge_model,
                              "questions": len(questions), "labels_applied": bool(labels), "variants": {}}
    for variant in variants:
        per = scored.get(variant.name, {})
        entry = {"description": variant.description, "run_errors": run_errors.get(variant.name, 0),
                 "snapshot": aggregate(list(per.get("snapshot", {}).values())),
                 "head_check": aggregate(list(per.get("head", {}).values())) if per.get("head") else None}
        if variant.name != reference and reference in scored:
            entry["vs_reference"] = compare(scored[reference].get("snapshot", {}), per.get("snapshot", {}))
        report["variants"][variant.name] = entry
    base = data_dir / project_name
    for name, file in (("dataset", "snapshots.json"), ("calibration", "calibration/calibration_result.json")):
        if (base / file).exists():
            report[name] = read_json(base / file)
    acl = sorted((base / "acl").glob("*.summary.json")) if (base / "acl").exists() else []
    report["acl"] = [read_json(p) for p in acl]
    write_json(base / "report.json", report)
    _write_csv(base / "metrics.csv", report)
    (base / "report.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def _write_csv(path: Path, report: Mapping[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["variant", "metric", "mean", "ci_low", "ci_high", "n", "diff_vs_reference",
                         "diff_ci_low", "diff_ci_high", "p_boot"])
        for name, entry in report["variants"].items():
            for metric, _, _ in METRICS:
                value = entry["snapshot"].get(metric) or {}
                diff = (entry.get("vs_reference") or {}).get(metric) or {}
                ci = value.get("ci") or (None, None)
                dci = diff.get("ci") or (None, None)
                writer.writerow([name, metric, value.get("mean"), ci[0], ci[1], value.get("n"), diff.get("diff"),
                                 dci[0], dci[1], diff.get("p_boot")])


def _diff(entry: Mapping[str, Any] | None, *, percent: bool = True) -> str:
    if not entry:
        return "n/a"
    scale = 100.0 if percent else 1.0
    low, high = entry["ci"]
    return f"{entry['diff'] * scale:+.1f} [{low * scale:+.1f}, {high * scale:+.1f}] p={entry['p_boot']:.3f}"


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [f"# SISU evaluation: {report['project']}", "",
             f"Questions after labels: {report['questions']} (labels applied: {report['labels_applied']}). "
             f"Judge: `{report['judge_model']}`. Reference variant: `{report['reference']}`.",
             "All rates are percentages with 95% bootstrap CIs (10,000 resamples, seed 7). "
             "Answer correctness is judged against the maintainer's reply.", ""]
    calibration = report.get("calibration")
    if calibration:
        c = calibration["correctness"]
        lines += [f"Judge check (correctness): kappa rater1-rater2 = {c['kappa_rater1_rater2']}, "
                  f"rater1-judge = {c['kappa_rater1_judge']}, rater2-judge = {c['kappa_rater2_judge']}, "
                  f"judge accuracy on {c['n_consensus']} consensus items = {c['judge_accuracy_on_consensus']}.", ""]
    else:
        lines += ["**Judge not yet checked against human raters.** Run `calibrate-export` and `calibrate` "
                  "before reporting correctness numbers.", ""]
    lines += ["## Quality (snapshot corpus)", "",
              "| variant | n | " + " | ".join(HEADLINE) + " |",
              "|---|---|" + "---|" * len(HEADLINE)]
    for name, entry in report["variants"].items():
        agg = entry["snapshot"]
        lines.append(f"| {name} | {agg['n']} | " + " | ".join(fmt_mean_ci(agg.get(m)) for m in HEADLINE) + " |")
    lines += ["", "## Cost and latency", "",
              "| variant | median s | p90 s | " + " | ".join(COST) + " | run errors |",
              "|---|---|---|" + "---|" * len(COST) + "---|"]
    for name, entry in report["variants"].items():
        agg = entry["snapshot"]
        median = f"{agg['latency_median_s']:.1f}" if agg.get("latency_median_s") is not None else "n/a"
        p90 = f"{agg['latency_p90_s']:.1f}" if agg.get("latency_p90_s") is not None else "n/a"
        lines.append(f"| {name} | {median} | {p90} | " +
                     " | ".join(fmt_mean_ci(agg.get(m), percent=False) for m in COST) + f" | {entry['run_errors']} |")
    lines += ["", f"## Paired differences vs `{report['reference']}` (percentage points)", "",
              "| variant | " + " | ".join(HEADLINE) + " |", "|---|" + "---|" * len(HEADLINE)]
    for name, entry in report["variants"].items():
        if "vs_reference" not in entry:
            continue
        lines.append(f"| {name} | " + " | ".join(_diff(entry["vs_reference"].get(m)) for m in HEADLINE) + " |")
    heads = {name: e["head_check"] for name, e in report["variants"].items() if e.get("head_check")}
    if heads:
        lines += ["", "## Questions unanswerable at their date, asked again on current docs", "",
                  "| variant | n | answer rate | correct (lenient) | evidence doc recall |", "|---|---|---|---|---|"]
        for name, agg in heads.items():
            lines.append(f"| {name} | {agg['n']} | {fmt_mean_ci(agg.get('answer_rate_answerable'))} | "
                         f"{fmt_mean_ci(agg.get('correct_lenient'))} | {fmt_mean_ci(agg.get('evidence_doc_recall'))} |")
    if report.get("acl"):
        lines += ["", "## Access control (half the documents allowed)", "",
                  "| variant | questions | answers citing forbidden docs | answers shown forbidden passages | "
                  "answer rate, gold allowed | answer rate, gold forbidden |", "|---|---|---|---|---|---|"]
        for item in report["acl"]:
            def pct(v):
                return "n/a" if v is None else f"{100 * v:.1f}"
            lines.append(f"| {item['variant']} | {item['questions']} | {item['answers_with_leaked_citations']} | "
                         f"{item['answers_with_leaked_presented_passages']} | {pct(item['answer_rate_gold_allowed'])} "
                         f"(n={item['n_gold_allowed']}) | {pct(item['answer_rate_gold_forbidden'])} "
                         f"(n={item['n_gold_forbidden']}) |")
    dataset = report.get("dataset")
    if dataset:
        lines += ["", "## Dataset", "",
                  f"Mined {dataset['questions_in']}, kept {dataset['questions_kept']} "
                  f"({dataset['answerable_at_snapshot']} answerable at their date, "
                  f"{dataset['unanswerable_at_snapshot']} not), {len(dataset['snapshots'])} doc snapshots, "
                  f"granularity {dataset['granularity']}. Drop reasons: {dataset['drop_reasons']}."]
    lines += ["", "## Variants", ""] + [f"- `{n}`: {e['description']}" for n, e in report["variants"].items()]
    return "\n".join(lines) + "\n"


def export_public(data_dir: Path, project_name: str) -> Path:
    """IDs, URLs, commits and gold paths only; no question or answer text."""

    labels = load_labels(data_dir, project_name)
    rows = []
    for q in read_jsonl(data_dir / project_name / "questions.jsonl"):
        label = labels.get(q["qid"], {})
        rows.append({"qid": q["qid"], "source": q["source"], "url": q["url"], "answer_url": q["answer_url"],
                     "created_at": q["created_at"], "snapshot_commit": q["snapshot_commit"],
                     "head_commit": q["head_commit"], "answer_commit": q["answer_commit"],
                     "gold": [{k: g[k] for k in ("path", "anchor", "url", "status_at_snapshot", "status_at_head")}
                              for g in q["gold"]],
                     "answerable_at_snapshot": label.get("answerable_at_snapshot") if label.get(
                         "answerable_at_snapshot") is not None else q["answerable_at_snapshot"],
                     "kept_after_review": label.get("keep", True)})
    path = data_dir / project_name / "public_release.jsonl"
    from .common import write_jsonl
    write_jsonl(path, rows)
    return path
