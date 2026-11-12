"""Command line for the SISU evaluation pipeline.

    python -m sisu_eval mine --project fastapi --limit 400
    python -m sisu_eval snapshot --project fastapi
    python -m sisu_eval review-export --project fastapi
    python -m sisu_eval review-import --project fastapi --csv eval_data/fastapi/review.csv
    python -m sisu_eval run --project fastapi --variants normal,thinking --head-check
    python -m sisu_eval judge --project fastapi
    python -m sisu_eval calibrate-export --project fastapi
    python -m sisu_eval calibrate --project fastapi --csv eval_data/fastapi/calibration/calibration_sheet.csv
    python -m sisu_eval acl --project fastapi
    python -m sisu_eval report --project fastapi
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from .common import DEFAULT_DATA_DIR, parse_time, write_jsonl


def _print(value) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m sisu_eval", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Where all evaluation data is written")
    sub = parser.add_subparsers(dest="command", required=True)

    def command(name, help_text):
        item = sub.add_parser(name, help=help_text)
        item.add_argument("--project", required=True, help="Name in sisu_eval/projects or a path to a project JSON")
        return item

    mine = command("mine", "Mine maintainer-answered questions with docs links")
    mine.add_argument("--limit", type=int, default=400, help="Maximum kept questions per source")
    mine.add_argument("--since", help="Only threads created on or after this date (YYYY-MM-DD)")
    mine.add_argument("--no-cache", action="store_true", help="Do not reuse cached API responses")

    snap = command("snapshot", "Resolve gold docs, assign dated snapshots, export doc trees")
    snap.add_argument("--granularity", choices=("day", "week", "month"), default="month",
                      help="Snapshot at the start of the question's day/week/month (default month: fewer snapshots to index)")

    review = command("review-export", "Write a CSV for a human pass over the mined set")
    review.add_argument("--out", type=Path)
    review_in = command("review-import", "Read the reviewed CSV into labels.json")
    review_in.add_argument("--csv", type=Path, required=True)

    index = command("index", "Index every snapshot used by the questions")
    index.add_argument("--variant", default="normal")

    run = command("run", "Answer every question with each variant")
    run.add_argument("--variants", help="Comma-separated names from sisu_eval/variants.json (default: all)")
    run.add_argument("--limit", type=int, help="Answer at most this many new questions per variant")
    run.add_argument("--head-check", action="store_true",
                     help="Also ask questions that were unanswerable at their date against current docs")
    run.add_argument("--retry-errors", action="store_true")

    judge = command("judge", "Judge correctness and citation support")
    judge.add_argument("--variants")
    judge.add_argument("--judge-model", default=None)

    cal_out = command("calibrate-export", "Blind sample for two human raters")
    cal_out.add_argument("--variants")
    cal_out.add_argument("--judge-model", default=None)
    cal_out.add_argument("--n", type=int, default=100)
    cal_in = command("calibrate", "Kappa between raters and the judge")
    cal_in.add_argument("--csv", type=Path, required=True)

    acl = command("acl", "Permission test: half the documents allowed")
    acl.add_argument("--variant", default="normal")
    acl.add_argument("--limit", type=int)

    report = command("report", "Write report.md, report.json and metrics.csv")
    report.add_argument("--variants")
    report.add_argument("--judge-model", default=None)

    status = command("status", "Progress, speed and health per variant")
    status.add_argument("--variants")
    status.add_argument("--check", help="Exit with code 3 if this variant looks broken (crashes, no answers, failed calls)")

    command("export-public", "IDs, URLs, commits and gold paths for a public release (no text)")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    data_dir = args.data_dir.resolve()
    from .projects import load_project
    project = load_project(args.project)

    if args.command == "mine":
        from .mine import mine_project
        cache = None if args.no_cache else data_dir / project.name / "api_cache"
        since = parse_time(args.since) if args.since else None
        records = mine_project(project, limit_per_source=args.limit, token=os.environ.get("GITHUB_TOKEN", ""),
                               since=since, cache_dir=cache)
        count = write_jsonl(data_dir / project.name / "mined.jsonl", records)
        _print({"mined": count, "path": str(data_dir / project.name / "mined.jsonl")})
        return 0

    if args.command == "snapshot":
        from .common import read_jsonl
        from .snapshot import build_snapshots, ensure_clone
        mined = read_jsonl(data_dir / project.name / "mined.jsonl")
        if not mined:
            print("error: no mined questions; run `mine` first", file=sys.stderr)
            return 2
        ensure_clone(project, data_dir)
        _print(build_snapshots(project, mined, data_dir, granularity=args.granularity))
        return 0

    if args.command == "review-export":
        from .labels import review_export
        out = args.out or data_dir / project.name / "review.csv"
        _print({"rows": review_export(data_dir, project.name, out), "csv": str(out),
                "columns_to_fill": ["keep (y/n)", "answerable_override (y/n or blank)", "note"]})
        return 0

    if args.command == "review-import":
        from .labels import review_import
        _print(review_import(data_dir, project.name, args.csv))
        return 0

    from .labels import load_labels
    from .runner import load_variants, select_variants
    from .snapshot import load_questions
    variants, reference = load_variants()
    questions = load_questions(project, data_dir, labels=load_labels(data_dir, project.name))

    if args.command == "index":
        from .runner import index_snapshot
        variant = variants[args.variant]
        commits = sorted({q["snapshot_commit"] for q in questions} | {q["head_commit"] for q in questions})
        _print([index_snapshot(data_dir, project.name, c, variant) for c in commits])
        return 0

    if args.command == "run":
        from .runner import run_variant
        summaries = [run_variant(data_dir, project.name, v, questions, head_check=args.head_check,
                                 retry_errors=args.retry_errors, limit=args.limit)
                     for v in select_variants(variants, args.variants)]
        _print(summaries)
        return 0

    from .judge import DEFAULT_JUDGE
    judge_model = getattr(args, "judge_model", None) or DEFAULT_JUDGE

    if args.command == "judge":
        from .judge import judge_variants
        names = [v.name for v in select_variants(variants, args.variants)]
        _print(judge_variants(data_dir, project.name, names, {q["qid"]: q for q in questions},
                              judge_model=judge_model))
        return 0

    if args.command == "calibrate-export":
        from .labels import calibration_export
        names = [v.name for v in select_variants(variants, args.variants)]
        _print(calibration_export(data_dir, project.name, judge_model, names, n=args.n))
        return 0

    if args.command == "calibrate":
        from .labels import calibrate
        _print(calibrate(data_dir, project.name, args.csv))
        return 0

    if args.command == "acl":
        from .acl import run_acl
        _print(run_acl(data_dir, project.name, variants[args.variant], questions, limit=args.limit))
        return 0

    if args.command == "report":
        from .report import build_report
        chosen = select_variants(variants, args.variants)
        report = build_report(data_dir, project.name, chosen, reference, judge_model, project=project)
        _print({"report": str(data_dir / project.name / "report.md"), "variants": list(report["variants"])})
        return 0

    if args.command == "status":
        from .runner import is_broken, variant_status
        rows = [variant_status(data_dir, project.name, v, questions) for v in select_variants(variants, args.variants)]
        print(f"{'variant':<20} {'done':>11} {'answered':>9} {'median s':>9} {'p90 s':>7} {'bad calls':>10} {'hours left':>11}")
        for r in rows:
            def num(value, digits=0):
                return "-" if value is None else f"{value:.{digits}f}"
            bad = "-" if r["bad_call_share"] is None else f"{r['bad_calls']} ({100 * r['bad_call_share']:.0f}%)"
            print(f"{r['variant']:<20} {r['done']:>5}/{r['expected']:<5} {r['answered']:>9} {num(r['median_s']):>9} "
                  f"{num(r['p90_s']):>7} {bad:>10} {num(r['hours_left_estimate'], 1):>11}")
        if args.check:
            if args.check not in variants:
                print(f"error: unknown variant {args.check!r}", file=sys.stderr)
                return 2
            reason = is_broken(variant_status(data_dir, project.name, variants[args.check], questions))
            if reason:
                print(f"STOP: variant {args.check} looks broken: {reason}", file=sys.stderr)
                return 3
        return 0

    if args.command == "export-public":
        from .report import export_public
        _print({"path": str(export_public(data_dir, project.name))})
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
