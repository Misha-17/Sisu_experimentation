"""Dated documentation snapshots and gold-evidence resolution.

Each question is answered against the docs as they stood at the start of its
time bucket (day/week/month), so documentation written later cannot leak in.
A gold link whose file or section did not exist yet makes the question
unanswerable at that snapshot; the same question is also kept for the
current (HEAD) snapshot to test that the answer appears once docs exist.
"""
from __future__ import annotations

from datetime import timedelta
import io
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
from typing import Any, Iterable

from .common import bucket_start, heading_slugs, iso, parse_time, read_jsonl, write_json, write_jsonl
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


def _gold_status(project: Project, git: Git, commit: str, path: str, anchor: str) -> str:
    if path not in git.files(commit):
        return "file_added_later"
    if anchor:
        text = git.text(commit, path) or ""
        if anchor not in heading_slugs(text):
            return "section_added_later"
    return "present"


def build_snapshots(project: Project, questions: list[dict[str, Any]], data_dir: Path, *,
                    granularity: str = "week", git: Git | None = None) -> dict[str, Any]:
    """Resolve gold evidence, assign dated snapshots and export their trees."""

    project_dir = data_dir / project.name
    git = git or Git(project_dir / "repo")
    branch = git.default_branch()
    head = git.head(branch)
    kept, dropped = [], []
    for question in questions:
        answered_at = iso(parse_time(question["answer_created_at"]) + timedelta(seconds=1))
        answer_commit = git.commit_before(branch, answered_at)
        if answer_commit is None:
            dropped.append({"qid": question["qid"], "reason": "answer_before_repo_history"})
            continue
        gold = []
        for link in question["doc_links"]:
            resolved = project.resolve_link(link, git.files(answer_commit))
            if resolved and resolved not in [(g["path"], g["anchor"]) for g in gold]:
                gold.append({"path": resolved[0], "anchor": resolved[1], "url": link})
        if not gold:
            dropped.append({"qid": question["qid"], "reason": "links_not_resolvable_to_docs_files"})
            continue
        start = bucket_start(parse_time(question["created_at"]), granularity)
        snapshot = git.commit_before(branch, iso(start))
        if snapshot is None:
            dropped.append({"qid": question["qid"], "reason": "question_before_repo_history"})
            continue
        for item in gold:
            item["status_at_snapshot"] = _gold_status(project, git, snapshot, item["path"], item["anchor"])
            item["status_at_head"] = _gold_status(project, git, head, item["path"], item["anchor"])
            item["indexed_path"] = project.indexed_name(item["path"])
        answerable = any(g["status_at_snapshot"] == "present" for g in gold)
        kept.append({**question, "bucket_start": iso(start), "granularity": granularity,
                     "snapshot_commit": snapshot, "answer_commit": answer_commit, "head_commit": head,
                     "gold": gold, "answerable_at_snapshot": answerable,
                     "answerable_at_head": any(g["status_at_head"] == "present" for g in gold)})
    snapshots_dir = project_dir / "snapshots"
    commits = sorted({q["snapshot_commit"] for q in kept} | ({head} if kept else set()))
    for commit in commits:
        export_snapshot(project, git, commit, snapshots_dir)
    summary = {"project": project.name, "branch": branch, "head_commit": head, "granularity": granularity,
               "questions_in": len(questions), "questions_kept": len(kept), "dropped": len(dropped),
               "answerable_at_snapshot": sum(q["answerable_at_snapshot"] for q in kept),
               "unanswerable_at_snapshot": sum(not q["answerable_at_snapshot"] for q in kept),
               "snapshots": commits, "drop_reasons": _count(d["reason"] for d in dropped)}
    write_jsonl(project_dir / "questions.jsonl", kept)
    write_jsonl(project_dir / "dropped.jsonl", dropped)
    write_json(project_dir / "snapshots.json", summary)
    return summary


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
