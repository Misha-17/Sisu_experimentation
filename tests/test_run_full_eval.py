"""The unattended runner script and the status/health check it relies on."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

from sisu_eval.__main__ import main as eval_main
from sisu_eval.common import write_jsonl
from sisu_eval.runner import is_broken, load_variants, variant_status
from sisu_eval.snapshot import load_questions

ROOT = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location("run_full_eval", ROOT / "scripts" / "run_full_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _questions(n=6):
    return [{"qid": f"toy:{i}", "answerable_at_snapshot": i != 0, "answerable_at_head": True,
             "snapshot_commit": "a" * 40, "head_commit": "b" * 40, "gold": []} for i in range(n)]


def _row(qid, *, answered=True, status="ok", wall=30.0, corpus="snapshot", error=None):
    row = {"variant": "normal", "qid": qid, "corpus": corpus, "answered": answered, "wall_s": wall,
           "model_calls": [{"role": "synthesis", "status": status, "thinking_chars": 0}]}
    if error:
        row["run_error"] = error
    return row


def _data(tmp_path, rows):
    data = tmp_path / "data"
    write_jsonl(data / "toy" / "questions.jsonl", _questions())
    write_jsonl(data / "toy" / "runs" / "normal.jsonl", rows)
    project_file = tmp_path / "toy.json"
    project_file.write_text(json.dumps({"name": "toy", "docs_repo": "unused", "docs_roots": ["docs"],
                                        "link_patterns": [{"regex": "^https://toy/(?P<path>.*)", "root": "docs"}],
                                        "sources": []}), encoding="utf-8")
    return data, project_file


def test_status_counts_progress_and_estimates_time(tmp_path):
    data, _ = _data(tmp_path, [_row(f"toy:{i}") for i in range(3)])
    variants, _ = load_variants()
    status = variant_status(data, "toy", variants["normal"], load_questions(SimpleNamespace(name="toy"), data))
    assert status["expected"] == 7  # 6 snapshot jobs + 1 head check for the unanswerable question
    assert status["done"] == 3 and status["answered"] == 3 and status["median_s"] == 30.0
    assert abs(status["hours_left_estimate"] - 4 * 30.0 / 3600) < 1e-9
    assert is_broken(status) is None  # fewer than 5 results: no verdict yet


def test_health_check_flags_broken_runs(tmp_path):
    variants, _ = load_variants()
    questions = _questions()
    cases = {
        "no question was answered": [_row(f"toy:{i}", answered=False) for i in range(6)],
        "of model calls failed": [_row(f"toy:{i}", status="error") for i in range(6)],
        "crashed": [_row(f"toy:{i}", error="ConnectionError: refused") for i in range(6)],
    }
    for expected, rows in cases.items():
        data, _ = _data(tmp_path / expected.replace(" ", "_"), rows)
        reason = is_broken(variant_status(data, "toy", variants["normal"], questions))
        assert reason and expected in reason
    data, _ = _data(tmp_path / "healthy", [_row(f"toy:{i}") for i in range(6)])
    assert is_broken(variant_status(data, "toy", variants["normal"], questions)) is None


def test_status_cli_exit_codes(tmp_path, capsys):
    data, project = _data(tmp_path, [_row(f"toy:{i}", answered=False) for i in range(6)])
    assert eval_main(["--data-dir", str(data), "status", "--project", str(project), "--variants", "normal"]) == 0
    assert "normal" in capsys.readouterr().out
    assert eval_main(["--data-dir", str(data), "status", "--project", str(project), "--check", "normal"]) == 3
    assert "STOP" in capsys.readouterr().err


def test_runner_order_in_dry_run(capsys):
    script = _script()
    assert script.main(["--dry-run", "--projects", "fastapi"]) == 0
    printed = [line.split("$ ", 1)[1] for line in capsys.readouterr().out.splitlines() if "$ " in line]
    joined = "\n".join(printed)
    order = ["ollama list", "pip install --upgrade pip", "torch==2.11.0", "download_encoder.py", "ollama pull gemma4:12b",
             "ollama pull gpt-oss:20b", "pytest tests", "--variants normal,thinking --limit 20",
             "status --project fastapi --variants normal --check normal",
             f"--variants {script.CORE} --head-check", f"judge --project fastapi --variants {script.CORE}",
             "calibrate-export --project fastapi", f"--variants {script.ABLATIONS}", "acl --project fastapi"]
    positions = [joined.find(item) for item in order]
    assert all(p >= 0 for p in positions), dict(zip(order, positions))
    assert positions == sorted(positions)
    # A report after the core phase, and a final one after the access-control test.
    assert joined.count("report --project fastapi") == 2
    assert joined.rfind("report --project fastapi") > joined.find("acl --project fastapi")


def test_runner_skip_setup_and_pilot_only(capsys):
    script = _script()
    assert script.main(["--dry-run", "--projects", "pydantic", "--skip-setup", "--pilot-only"]) == 0
    printed = [line for line in capsys.readouterr().out.splitlines() if "$ " in line]
    assert not any("pip install" in line or "ollama pull" in line for line in printed)
    assert not any("judge" in line for line in printed)
    assert any("--limit 20" in line for line in printed)


def test_results_zip_has_only_results(tmp_path):
    script = _script()
    project = tmp_path / "eval_data" / "toy"
    for name in ("runs/normal.jsonl", "judgments/j.jsonl", "calibration/calibration_sheet.csv", "report.md",
                 "questions.jsonl", "repo/HEAD", "snapshots/abc/tree/docs/a.md", "workspaces/eval/x/reader.sqlite3",
                 "api_cache/k.json"):
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    target = script.save_results(tmp_path, "toy", "core", stamp="20270101-0000")
    names = set(zipfile.ZipFile(target).namelist())
    assert "eval_data/toy/runs/normal.jsonl" in names and "eval_data/toy/report.md" in names
    assert "eval_data/toy/calibration/calibration_sheet.csv" in names
    assert not any(part in name for name in names for part in ("/repo/", "/snapshots/", "/workspaces/", "/api_cache/"))
