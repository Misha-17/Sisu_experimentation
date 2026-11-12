"""Run the whole SISU evaluation on this machine's GPU, in priority order.

Usage (Windows):  py -3.11 scripts\\run_full_eval.py
Usage (Linux):    python3.11 scripts/run_full_eval.py

Safe to stop (Ctrl+C) and rerun at any time: every step skips work that is
already done. Output is logged to logs/, and after each phase a results zip
is written to results/ for sending back.

Order:
  0. Setup: .venv, packages, E5 encoder, Ollama models, test suite.
  1. Pilot: 20 questions with `normal` and `thinking` on the first project;
     stops if `normal` looks broken.
  2. Per project: core variants -> judge -> rating sheet -> report -> zip,
     then ablations -> judge -> access-control test -> report -> zip.
Only the Python standard library is used here, so it runs before .venv exists.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ("fastapi", "pydantic", "cpython-devguide")
CORE = "normal,thinking,normal_long_budget,plain_rag"
ABLATIONS = "normal_e4b,lexical_only,dense_only,no_planning,no_review,no_neighbors"
MODELS = ("gemma4:12b", "gemma4:e4b", "gpt-oss:20b")
RESULT_ITEMS = ("questions.jsonl", "snapshots.json", "labels.json", "dropped.jsonl", "runs", "judgments",
                "calibration", "acl", "report.md", "report.json", "metrics.csv")


class StepFailed(RuntimeError):
    pass


def venv_python(root: Path = ROOT) -> Path:
    if os.name == "nt":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


class Runner:
    def __init__(self, root: Path, *, dry_run: bool = False, log_path: Path | None = None):
        self.root = root
        self.dry_run = dry_run
        self.commands: list[list[str]] = []
        self.log = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log = log_path.open("a", encoding="utf-8")

    def say(self, message: str) -> None:
        line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
        print(line, flush=True)
        if self.log:
            self.log.write(line + "\n")
            self.log.flush()

    def step(self, message: str) -> None:
        self.say("")
        self.say(f"==> {message}")

    def run(self, command: list[str], *, check: bool = True) -> int:
        command = [str(part) for part in command]
        self.commands.append(command)
        self.say("$ " + " ".join(command))
        if self.dry_run:
            return 0
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        process = subprocess.Popen(command, cwd=self.root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", env=env)
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if self.log:
                self.log.write(line)
                self.log.flush()
        code = process.wait()
        if check and code != 0:
            raise StepFailed(f"command failed with exit code {code}: {' '.join(command)}")
        return code

    def eval(self, *arguments: str, check: bool = True) -> int:
        return self.run([str(venv_python(self.root)), "-m", "sisu_eval", *arguments], check=check)

    def close(self) -> None:
        if self.log:
            self.log.close()


def keep_awake() -> None:
    """Stop Windows from sleeping while this process runs (screen may still turn off)."""

    if os.name != "nt":
        return
    import ctypes

    es_continuous, es_system_required = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)


def has_questions(root: Path, project: str) -> bool:
    return (root / "eval_data" / project / "questions.jsonl").is_file()


def save_results(root: Path, project: str, phase: str, *, stamp: str | None = None) -> Path | None:
    source = root / "eval_data" / project
    stamp = stamp or datetime.now().strftime("%Y%m%d-%H%M")
    target = root / "results" / f"{project}-{phase}-{stamp}.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    files = []
    for name in RESULT_ITEMS:
        path = source / name
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(p for p in sorted(path.rglob("*")) if p.is_file())
    if not files:
        return None
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, Path("eval_data") / project / path.relative_to(source))
    return target


def setup(runner: Runner) -> None:
    runner.step("Checking tools")
    for tool in ("git", "ollama"):
        if shutil.which(tool) is None and not runner.dry_run:
            raise StepFailed(f"{tool} is not installed. See RUNNING.md, step 1.")
    if runner.run(["ollama", "list"], check=False) != 0:
        raise StepFailed("Ollama is not running. Open the Ollama app (or run `ollama serve`), then rerun this script.")
    python = venv_python(runner.root)
    if not python.exists():
        if sys.version_info < (3, 11):
            raise StepFailed(f"Python 3.11 or newer is needed to create .venv; this is {sys.version.split()[0]}. "
                             "Run the script with `py -3.11` on Windows.")
        runner.step("Creating .venv")
        runner.run([sys.executable, "-m", "venv", str(runner.root / ".venv")])
    runner.step("Installing Python packages (first time takes a few minutes)")
    runner.run([python, "-m", "pip", "install", "--upgrade", "pip"])
    runner.run([python, "-m", "pip", "install", "torch==2.11.0", "--index-url", "https://download.pytorch.org/whl/cpu"])
    runner.run([python, "-m", "pip", "install", "-r", "requirements-dev.txt"])
    runner.step("Installing the E5 retrieval encoder (skips verified files)")
    runner.run([python, "scripts/download_encoder.py"])
    runner.step("Pulling Ollama models (about 30 GB the first time)")
    for model in MODELS:
        runner.run(["ollama", "pull", model])
    runner.step("Running the test suite")
    runner.run([python, "-m", "pytest", "tests", "-q"])


def pilot(runner: Runner, project: str) -> None:
    runner.step(f"Pilot on {project}: 20 questions each for normal and thinking")
    runner.eval("index", "--project", project)
    runner.eval("run", "--project", project, "--variants", "normal,thinking", "--limit", "20")
    runner.eval("status", "--project", project, "--variants", "normal,thinking")
    runner.run(["ollama", "ps"], check=False)
    if runner.eval("status", "--project", project, "--variants", "normal", "--check", "normal", check=False) == 3:
        raise StepFailed("The normal variant looks broken in the pilot (see the STOP line above). "
                         "Send the latest log in logs/ to Shivansh before running anything else.")


def run_project(runner: Runner, project: str) -> list[Path]:
    saved = []
    data = runner.root / "eval_data" / project
    runner.step(f"[{project}] indexing")
    runner.eval("index", "--project", project)
    runner.step(f"[{project}] core variants: {CORE}")
    runner.eval("run", "--project", project, "--variants", CORE, "--head-check")
    runner.step(f"[{project}] judging core variants")
    runner.eval("judge", "--project", project, "--variants", CORE)
    if not (data / "calibration" / "calibration_sheet.csv").exists():
        runner.eval("calibrate-export", "--project", project, "--variants", CORE, "--n", "100")
    runner.eval("report", "--project", project)
    runner.eval("status", "--project", project)
    if not runner.dry_run:
        path = save_results(runner.root, project, "core")
        if path:
            runner.say(f"Results saved: {path}  <- send this file to Shivansh")
            saved.append(path)
    runner.step(f"[{project}] ablations: {ABLATIONS}")
    runner.eval("run", "--project", project, "--variants", ABLATIONS)
    runner.step(f"[{project}] judging all variants")
    runner.eval("judge", "--project", project)
    runner.step(f"[{project}] access-control test")
    runner.eval("acl", "--project", project)
    runner.eval("report", "--project", project)
    runner.eval("status", "--project", project)
    if not runner.dry_run:
        path = save_results(runner.root, project, "all")
        if path:
            runner.say(f"Results saved: {path}  <- send this file to Shivansh")
            saved.append(path)
    return saved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--projects", default=",".join(PROJECTS), help="Comma-separated, in run order")
    parser.add_argument("--skip-setup", action="store_true", help="Skip installs, downloads and tests")
    parser.add_argument("--pilot-only", action="store_true", help="Stop after the 20-question pilot")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without running them")
    args = parser.parse_args(argv)
    projects = [p.strip() for p in args.projects.split(",") if p.strip()]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = None if args.dry_run else ROOT / "logs" / f"run_{stamp}.log"
    runner = Runner(ROOT, dry_run=args.dry_run, log_path=log_path)
    keep_awake()
    try:
        if not args.skip_setup:
            setup(runner)
        available = [p for p in projects if has_questions(ROOT, p)] if not args.dry_run else projects
        missing = [p for p in projects if p not in available]
        for project in missing:
            runner.say(f"Skipping {project}: eval_data/{project}/questions.jsonl not found")
        if not available:
            raise StepFailed("No questions found under eval_data. Unzip the eval_data zip from Shivansh into "
                             f"{ROOT} first (RUNNING.md, step 3).")
        pilot(runner, available[0])
        if args.pilot_only:
            runner.step("Pilot finished. Send the status table above to Shivansh.")
            return 0
        saved = []
        for project in available:
            saved += run_project(runner, project)
        runner.step("All done. Send these files to Shivansh:")
        for path in saved:
            runner.say(f"  {path}")
        return 0
    except StepFailed as exc:
        runner.say(f"STOPPED: {exc}")
        if log_path:
            runner.say(f"Log: {log_path}. Fix the problem and rerun the same command; finished work is kept.")
        return 1
    except KeyboardInterrupt:
        runner.say("Interrupted. Rerun the same command to continue where it stopped.")
        return 130
    finally:
        runner.close()


if __name__ == "__main__":
    raise SystemExit(main())
