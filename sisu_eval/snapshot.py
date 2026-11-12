"""Dated documentation snapshots and gold-evidence resolution.

Each question is answered against the docs as they stood at the start of its
time bucket (day/week/month; month by default), so documentation written later cannot leak in.
A gold link whose file or section did not exist yet makes the question
unanswerable at that snapshot; the same question is also kept for the
current (HEAD) snapshot to test that the answer appears once docs exist.
"""
from __future__ import annotations

from datetime import timedelta
import io
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
from typing import Any, Iterable

from .common import anchor_title, bucket_start, heading_slugs, iso, parse_time, read_jsonl, write_json, write_jsonl
from .projects import Project


class Git:
    def __init__(self, repo: Path):
        self.repo = Path(repo)
        self._files: dict[str, frozenset[str]] = {}
        self._texts: dict[tuple[str, str], str | None] = {}

    def run(self, *args: str, binary: bool = False) -> Any:
        result = subprocess.run(["git", "-C", str(self.repo), *args], capture_output=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.decode('utf-8', 'replace').strip()}")
        return result.stdout if binary else result.stdout.decode("utf-8", "replace").strip()

    def default_branch(self) -> str:
        try:
            return self.run("symbolic-ref", "--short", "refs/remotes/origin/HEAD")
        except RuntimeError:
            return self.run("rev-parse", "--abbrev-ref", "HEAD")

    def head(self, branch: str) -> str:
        return self.run("rev-parse", branch)

    def commit_before(self, branch: str, moment_iso: str) -> str | None:
        value = self.run("rev-list", "-1", f"--before={moment_iso}", branch)
        return value or None

    def files(self, commit: str) -> frozenset[str]:
        if commit not in self._files:
            listing = self.run("ls-tree", "-r", "--name-only", "-z", commit, binary=True)
            self._files[commit] = frozenset(x.decode("utf-8") for x in listing.split(b"\0") if x)
        return self._files[commit]

    def text(self, commit: str, path: str) -> str | None:
        key = (commit, path)
        if key not in self._texts:
            try:
                self._texts[key] = self.run("show", f"{commit}:{path}", binary=True).decode("utf-8", "replace")
            except RuntimeError:
                self._texts[key] = None
        return self._texts[key]

    def archive(self, commit: str, paths: Iterable[str]) -> bytes:
        return self.run("archive", "--format=tar", commit, "--", *paths, binary=True)


def ensure_clone(project: Project, data_dir: Path) -> Path:
    repo = data_dir / project.name / "repo"
    if (repo / ".git").exists() or (repo / "HEAD").exists():
        subprocess.run(["git", "-C", str(repo), "fetch", "--quiet", "origin"], check=True)
    else:
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", "--no-checkout", project.docs_repo, str(repo)], check=True)
    return repo


def _safe_extract(archive: bytes, destination: Path) -> list[str]:
    names = []
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise RuntimeError(f"Unsafe path in git archive: {member.name}")
            if member.issym() or member.islnk():
                continue
            if member.isfile():
                target = destination / Path(*path.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                target.write_bytes(source.read() if source else b"")
                names.append(str(path))
    return names


def export_snapshot(project: Project, git: Git, commit: str, snapshots_dir: Path) -> Path:
    """Write the docs tree at ``commit`` to ``snapshots_dir/<commit>/tree``."""

    root = snapshots_dir / commit
    tree = root / "tree"
    manifest = root / "manifest.json"
    if manifest.exists():
        return tree
    if root.exists():
        shutil.rmtree(root)
    tree.mkdir(parents=True)
    files = git.files(commit)
    roots = [r for r in project.docs_roots if r == "" or any(f.startswith(r.rstrip("/") + "/") for f in files)]
    if not roots:
        raise RuntimeError(f"No docs roots {project.docs_roots} exist at commit {commit}")
    archive = git.archive(commit, [r or "." for r in roots])
    mapping = {}
    for name in _safe_extract(archive, tree):
        path = tree / name
        if not project.is_doc_path(name):
            path.unlink()
            continue
        indexed = project.indexed_name(name)
        if indexed != name:
            path.rename(tree / indexed)
        mapping[indexed] = name
    write_json(manifest, {"commit": commit, "files": len(mapping), "indexed_to_repo_path": mapping})
    return tree


def has_primary_root(project: Project, git: Git, commit: str) -> bool:
    """True when the first docs root exists at ``commit``.

    Projects move their docs (FastAPI moved them to docs/en/docs when it added
    translations). Questions whose snapshot predates the configured layout are
    dropped rather than answered against a tree with no docs in it.
    """

    primary = project.docs_roots[0] if project.docs_roots else ""
    if primary == "":
        return True
    prefix = primary.rstrip("/") + "/"
    return any(path.startswith(prefix) for path in git.files(commit))


def anchor_verified(git: Git, answer_commit: str, path: str, anchor: str) -> bool:
    """True when ``anchor`` is a heading in the source file at answer time."""

    return bool(anchor) and anchor in heading_slugs(git.text(answer_commit, path) or "")


def _gold_status(project: Project, git: Git, commit: str, path: str, anchor: str, *,
                 answer_commit: str | None = None) -> str:
    """present / file_added_later / section_added_later for one gold link.

    A section counts as present if its anchor exists, or if a heading with the
    same title exists (anchor syntax and slug rules change over time, for
    example FastAPI's switch to explicit ``{ #id }`` anchors in 2025-08).
    """

    if path not in git.files(commit):
        return "file_added_later"
    if anchor and answer_commit and not anchor_verified(git, answer_commit, path, anchor):
        # Anchors generated at build time (API reference entries such as
        # #pydantic.main.BaseModel) are not headings in the source file, so
        # their existence cannot be dated. Fall back to page-level existence.
        return "present"
    if anchor:
        slugs = heading_slugs(git.text(commit, path) or "")
        if anchor in slugs:
            return "present"
        title = anchor_title(git.text(answer_commit, path) or "", anchor) if answer_commit else None
        if title and title in slugs:
            return "present"
        return "section_added_later"
    return "present"


def build_snapshots(project: Project, questions: list[dict[str, Any]], data_dir: Path, *,
                    granularity: str = "month", git: Git | None = None) -> dict[str, Any]:
    """Resolve gold evidence, assign dated snapshots and export their trees."""

    project_dir = data_dir / project.name
    git = git or Git(project_dir / "repo")
    branch = git.default_branch()
    head = git.head(branch)
    kept, dropped = [], []
    for question in questions:
        def drop(reason: str) -> None:
            dropped.append({"qid": question["qid"], "created_at": question["created_at"], "reason": reason})

        answered_at = iso(parse_time(question["answer_created_at"]) + timedelta(seconds=1))
        answer_commit = git.commit_before(branch, answered_at)
        if answer_commit is None:
            drop("answer_before_repo_history")
            continue
        gold = []
        for link in question["doc_links"]:
            resolved = project.resolve_link(link, git.files(answer_commit))
            if resolved and resolved not in [(g["path"], g["anchor"]) for g in gold]:
                gold.append({"path": resolved[0], "anchor": resolved[1], "url": link})
        if not gold:
            drop("links_not_resolvable_to_docs_files")
            continue
        start = bucket_start(parse_time(question["created_at"]), granularity)
        snapshot = git.commit_before(branch, iso(start))
        if snapshot is None:
            drop("question_before_repo_history")
            continue
        if not has_primary_root(project, git, snapshot):
            drop("docs_layout_differs_at_question_date")
            continue

        def statuses(commit: str) -> list[str]:
            return [_gold_status(project, git, commit, g["path"], g["anchor"], answer_commit=answer_commit)
                    for g in gold]

        at_snapshot = statuses(snapshot)
        snapshot_reason = "bucket_start"
        if "present" not in at_snapshot:
            # The docs may have changed between the bucket start and the
            # question itself; then the bucket snapshot would wrongly make the
            # question unanswerable. Use the exact question time instead.
            exact = git.commit_before(branch, iso(parse_time(question["created_at"])))
            if exact and exact != snapshot and has_primary_root(project, git, exact):
                at_exact = statuses(exact)
                if "present" in at_exact:
                    snapshot, at_snapshot, snapshot_reason = exact, at_exact, "exact_question_time"
        for item, status in zip(gold, at_snapshot):
            item["status_at_snapshot"] = status
            item["status_at_head"] = _gold_status(project, git, head, item["path"], item["anchor"],
                                                  answer_commit=answer_commit)
            item["indexed_path"] = project.indexed_name(item["path"])
            item["anchor_verified"] = anchor_verified(git, answer_commit, item["path"], item["anchor"])
        answerable = any(g["status_at_snapshot"] == "present" for g in gold)
        days = (parse_time(question["answer_created_at"]) - parse_time(question["created_at"])).days
        kept.append({**question, "bucket_start": iso(start), "granularity": granularity,
                     "snapshot_commit": snapshot, "snapshot_reason": snapshot_reason, "days_to_answer": days,
                     "answer_commit": answer_commit, "head_commit": head,
                     "gold": gold, "answerable_at_snapshot": answerable,
                     "answerable_at_head": any(g["status_at_head"] == "present" for g in gold)})
    snapshots_dir = project_dir / "snapshots"
    commits = sorted({q["snapshot_commit"] for q in kept} | ({head} if kept else set()))
    for commit in commits:
        export_snapshot(project, git, commit, snapshots_dir)
    # Remove snapshot trees left over from earlier runs (derived from git; they
    # would only inflate the folder that is sent to the GPU machine).
    if snapshots_dir.exists():
        for old in snapshots_dir.iterdir():
            if old.is_dir() and re.fullmatch(r"[0-9a-f]{40}", old.name) and old.name not in commits:
                shutil.rmtree(old)
    unanswerable = [q for q in kept if not q["answerable_at_snapshot"]]
    summary = {"project": project.name, "branch": branch, "head_commit": head, "granularity": granularity,
               "questions_in": len(questions), "questions_kept": len(kept), "dropped": len(dropped),
               "answerable_at_snapshot": sum(q["answerable_at_snapshot"] for q in kept),
               "unanswerable_at_snapshot": sum(not q["answerable_at_snapshot"] for q in kept),
               "snapshots": commits, "drop_reasons": _count(d["reason"] for d in dropped),
               "kept_by_year": dict(sorted(_count(q["created_at"][:4] for q in kept).items())),
               "dropped_by_year": dict(sorted(_count(d["created_at"][:4] for d in dropped).items())),
               "kept_date_range": [min(q["created_at"] for q in kept), max(q["created_at"] for q in kept)]
               if kept else None,
               "unanswerable_because": _count("+".join(sorted({g["status_at_snapshot"] for g in q["gold"]}))
                                              for q in unanswerable),
               "anchors_not_verifiable": sum(1 for q in kept for g in q["gold"]
                                             if g["anchor"] and not g["anchor_verified"]),
               "exact_time_snapshots": sum(1 for q in kept if q["snapshot_reason"] == "exact_question_time"),
               "median_days_to_answer": {
                   "answerable": _median([q["days_to_answer"] for q in kept if q["answerable_at_snapshot"]]),
                   "unanswerable": _median([q["days_to_answer"] for q in unanswerable])},
               "answerable_at_head": sum(1 for q in kept if q["answerable_at_head"])}
    write_jsonl(project_dir / "questions.jsonl", kept)
    write_jsonl(project_dir / "dropped.jsonl", dropped)
    write_json(project_dir / "snapshots.json", summary)
    return summary


def _median(values: list[int]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    return float(ordered[middle]) if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def _count(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def load_questions(project: Project, data_dir: Path, *, labels: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    rows = read_jsonl(data_dir / project.name / "questions.jsonl")
    if labels:
        rows = [r for r in rows if labels.get(r["qid"], {}).get("keep", True)]
        for row in rows:
            override = labels.get(row["qid"], {}).get("answerable_at_snapshot")
            if override is not None:
                row["answerable_at_snapshot"] = bool(override)
    return rows
