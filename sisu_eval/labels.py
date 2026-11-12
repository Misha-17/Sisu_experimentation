"""Human review of the mined set, and blind two-rater calibration of the judge."""
from __future__ import annotations

import csv
from pathlib import Path
import random
from typing import Any, Mapping

from .common import read_json, read_jsonl, write_json
from .judge import judgments_path
from .metrics import cohen_kappa
from .runner import results_path

TRUE = {"y", "yes", "1", "true", "keep", "k"}
FALSE = {"n", "no", "0", "false", "drop", "d"}
CORRECTNESS = ("correct", "partially_correct", "incorrect")
SUPPORT = ("fully_supported", "partially_supported", "unsupported")


def _flag(value: str) -> bool | None:
    value = (value or "").strip().casefold()
    if value in TRUE:
        return True
    if value in FALSE:
        return False
    if not value:
        return None
    raise ValueError(f"Expected y/n, got {value!r}")


# Dataset review -------------------------------------------------------------------

def review_export(data_dir: Path, project_name: str, out_csv: Path) -> int:
    rows = read_jsonl(data_dir / project_name / "questions.jsonl")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["qid", "url", "created_at", "answerable_at_snapshot", "gold", "question",
                         "maintainer_answer", "keep", "answerable_override", "note"])
        for row in rows:
            gold = "; ".join(f"{g['path']}#{g['anchor']} [{g['status_at_snapshot']}]" if g["anchor"] else
                             f"{g['path']} [{g['status_at_snapshot']}]" for g in row["gold"])
            writer.writerow([row["qid"], row["url"], row["created_at"], "y" if row["answerable_at_snapshot"] else "n",
                             gold, row["question"][:800], row["answer"][:1200], "y", "", ""])
    return len(rows)


def review_import(data_dir: Path, project_name: str, csv_path: Path) -> dict[str, Any]:
    labels = {}
    with csv_path.open("r", newline="", encoding="utf-8-sig") as stream:
        for line, row in enumerate(csv.DictReader(stream), 2):
            try:
                keep = _flag(row.get("keep", ""))
                override = _flag(row.get("answerable_override", ""))
            except ValueError as exc:
                raise ValueError(f"{csv_path}:{line}: {exc}") from exc
            labels[row["qid"]] = {"keep": True if keep is None else keep, "answerable_at_snapshot": override,
                                  "note": row.get("note", "")}
    path = data_dir / project_name / "labels.json"
    write_json(path, labels)
    return {"labels": len(labels), "dropped": sum(1 for x in labels.values() if not x["keep"]),
            "overrides": sum(1 for x in labels.values() if x["answerable_at_snapshot"] is not None),
            "path": str(path)}


def load_labels(data_dir: Path, project_name: str) -> dict[str, dict[str, Any]]:
    path = data_dir / project_name / "labels.json"
    return read_json(path) if path.exists() else {}


# Judge calibration -------------------------------------------------------------------

def calibration_export(data_dir: Path, project_name: str, judge_model: str, variants: list[str], *,
                       n: int = 100, seed: int = 7) -> dict[str, Any]:
    questions = {q["qid"]: q for q in read_jsonl(data_dir / project_name / "questions.jsonl")}
    judged = {(r["variant"], r["qid"], r["corpus"]): r for r in read_jsonl(judgments_path(data_dir, project_name, judge_model))
              if not r.get("auto") and r["correctness"]["verdict"] in CORRECTNESS}
    results = {}
    for variant in variants:
        for row in read_jsonl(results_path(data_dir, project_name, variant)):
            results[(row["variant"], row["qid"], row["corpus"])] = row
    keys = [k for k in judged if k in results and k[1] in questions]
    strata: dict[tuple[str, str], list] = {}
    for key in keys:
        strata.setdefault((key[0], judged[key]["correctness"]["verdict"]), []).append(key)
    rng = random.Random(seed)
    for bucket in strata.values():
        rng.shuffle(bucket)
    sample, order = [], sorted(strata)
    while len(sample) < min(n, len(keys)):
        for bucket_key in order:
            if strata[bucket_key] and len(sample) < n:
                sample.append(strata[bucket_key].pop())
    rng.shuffle(sample)
    base = data_dir / project_name / "calibration"
    base.mkdir(parents=True, exist_ok=True)
    sheet, key_file = base / "calibration_sheet.csv", base / "calibration_key.json"
    key_rows = {}
    with sheet.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["item", "question", "maintainer_answer", "system_answer", "cited_passages",
                         "rater_1_correctness", "rater_2_correctness", "rater_1_support", "rater_2_support"])
        for index, key in enumerate(sample, 1):
            item = f"I{index:03d}"
            row, question = results[key], questions[key[1]]
            passages = "\n---\n".join(f"[{c.get('path')}] {c.get('quote', '')[:600]}" for c in row.get("citations", [])[:6])
            writer.writerow([item, question["question"][:1500], question["answer"][:2000], row["text"][:3000],
                             passages, "", "", "", ""])
            key_rows[item] = {"variant": key[0], "qid": key[1], "corpus": key[2],
                              "judge_correctness": judged[key]["correctness"]["verdict"],
                              "judge_support": ((judged[key].get("support") or {}).get("verdict"))}
    write_json(key_file, {"judge_model": judge_model, "allowed_correctness": CORRECTNESS,
                          "allowed_support": SUPPORT, "items": key_rows})
    return {"items": len(sample), "sheet": str(sheet), "key": str(key_file)}


def calibrate(data_dir: Path, project_name: str, sheet_csv: Path) -> dict[str, Any]:
    base = data_dir / project_name / "calibration"
    key = read_json(base / "calibration_key.json")["items"]
    columns = {name: [] for name in ("r1c", "r2c", "jc", "r1s", "r2s", "js")}
    with sheet_csv.open("r", newline="", encoding="utf-8-sig") as stream:
        for line, row in enumerate(csv.DictReader(stream), 2):
            item = key.get(row["item"])
            if item is None:
                raise ValueError(f"{sheet_csv}:{line}: unknown item {row['item']!r}")
            values = {"r1c": row["rater_1_correctness"], "r2c": row["rater_2_correctness"],
                      "r1s": row["rater_1_support"], "r2s": row["rater_2_support"]}
            for name, value in values.items():
                value = value.strip().casefold()
                allowed = CORRECTNESS if name.endswith("c") else SUPPORT
                if value and value not in allowed:
                    raise ValueError(f"{sheet_csv}:{line}: {name} must be one of {allowed}, got {value!r}")
                columns[name].append(value)
            columns["jc"].append(item["judge_correctness"] or "")
            columns["js"].append(item["judge_support"] if item["judge_support"] in SUPPORT else "")

    def block(r1, r2, judge):
        consensus = [(a, j) for a, b, j in zip(r1, r2, judge) if a and a == b and j]
        return {"n_rated_by_both": sum(1 for a, b in zip(r1, r2) if a and b),
                "kappa_rater1_rater2": cohen_kappa(r1, r2),
                "kappa_rater1_judge": cohen_kappa(r1, judge), "kappa_rater2_judge": cohen_kappa(r2, judge),
                "judge_accuracy_on_consensus": (sum(1 for a, j in consensus if a == j) / len(consensus))
                if consensus else None, "n_consensus": len(consensus)}

    result = {"correctness": block(columns["r1c"], columns["r2c"], columns["jc"]),
              "support": block(columns["r1s"], columns["r2s"], columns["js"])}
    write_json(base / "calibration_result.json", result)
    return result
