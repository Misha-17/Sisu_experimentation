"""Project configuration and documentation-link resolution."""
from __future__ import annotations

from dataclasses import dataclass, field
import fnmatch
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable, Mapping

from .common import PACKAGE_DIR, read_json, url_without_fragment

PROJECTS_DIR = PACKAGE_DIR / "projects"
DEFAULT_CANDIDATES = ("{p}.md", "{p}/index.md", "{p}.rst", "{p}/index.rst", "{p}.txt", "{p}")
DEFAULT_ASSOCIATIONS = ("OWNER", "MEMBER", "COLLABORATOR")


@dataclass(frozen=True)
class LinkPattern:
    regex: re.Pattern[str]
    root: str


@dataclass(frozen=True)
class Project:
    name: str
    docs_repo: str
    docs_roots: tuple[str, ...]
    link_patterns: tuple[LinkPattern, ...]
    sources: tuple[Mapping[str, Any], ...]
    path_candidates: tuple[str, ...] = DEFAULT_CANDIDATES
    convert_to_txt: tuple[str, ...] = (".rst", ".mdx")
    exclude_globs: tuple[str, ...] = ()
    maintainers: frozenset[str] = frozenset()
    maintainer_associations: frozenset[str] = frozenset(DEFAULT_ASSOCIATIONS)
    min_question_chars: int = 40
    drop_if_question_has_gold_link: bool = True
    trust_accepted_answers: bool = False
    description: str = ""
    raw: Mapping[str, Any] = field(default_factory=dict)

    # -- link resolution ---------------------------------------------------
    def resolve_link(self, url: str, files: Iterable[str] | set[str]) -> tuple[str, str] | None:
        """Map a documentation URL to ``(repo_path, anchor)`` if that file exists.

        ``files`` is the set of repository paths at the commit being checked.
        Only paths under a configured docs root are accepted.
        """

        files = files if isinstance(files, (set, frozenset)) else set(files)
        base, anchor = url_without_fragment(url.strip())
        for pattern in self.link_patterns:
            match = pattern.regex.match(base)
            if not match:
                continue
            raw = match.group("path").strip("/")
            variants = [raw]
            if raw.endswith(".html"):
                variants.append(raw[:-5])
            if raw in {"", "index.html"}:
                variants.append("index")
            for value in variants:
                for template in self.path_candidates:
                    candidate = template.format(p=value).strip("/")
                    joined = str(PurePosixPath(pattern.root) / candidate) if pattern.root else candidate
                    joined = joined.replace("/./", "/")
                    if joined in files and self.is_doc_path(joined):
                        return joined, anchor.casefold()
        return None

    def is_doc_path(self, path: str) -> bool:
        if not any(root == "" or path == root or path.startswith(root.rstrip("/") + "/") for root in self.docs_roots):
            return False
        return not any(fnmatch.fnmatch(path, pattern) for pattern in self.exclude_globs)

    def indexed_name(self, path: str) -> str:
        """Name the file gets inside an exported snapshot."""

        suffix = PurePosixPath(path).suffix.casefold()
        return path + ".txt" if suffix in self.convert_to_txt else path


def load_project(name_or_path: str) -> Project:
    path = Path(name_or_path)
    if not path.exists():
        path = PROJECTS_DIR / f"{name_or_path}.json"
    if not path.exists():
        available = sorted(p.stem for p in PROJECTS_DIR.glob("*.json"))
        raise FileNotFoundError(f"Unknown project {name_or_path!r}; available: {', '.join(available)}")
    return project_from_mapping(read_json(path))


def project_from_mapping(data: Mapping[str, Any]) -> Project:
    patterns = tuple(LinkPattern(re.compile(item["regex"]), str(item.get("root", "")).strip("/"))
                     for item in data["link_patterns"])
    for pattern in patterns:
        if "path" not in pattern.regex.groupindex:
            raise ValueError(f"link pattern {pattern.regex.pattern!r} needs a named group 'path'")
    return Project(
        name=str(data["name"]),
        docs_repo=str(data["docs_repo"]),
        docs_roots=tuple(str(x).strip("/") for x in data.get("docs_roots", [""])),
        link_patterns=patterns,
        sources=tuple(data["sources"]),
        path_candidates=tuple(data.get("path_candidates", DEFAULT_CANDIDATES)),
        convert_to_txt=tuple(x.casefold() for x in data.get("convert_to_txt", (".rst", ".mdx"))),
        exclude_globs=tuple(data.get("exclude_globs", ())),
        maintainers=frozenset(x.casefold() for x in data.get("maintainers", ())),
        maintainer_associations=frozenset(data.get("maintainer_associations", DEFAULT_ASSOCIATIONS)),
        min_question_chars=int(data.get("min_question_chars", 40)),
        drop_if_question_has_gold_link=bool(data.get("drop_if_question_has_gold_link", True)),
        trust_accepted_answers=bool(data.get("trust_accepted_answers", False)),
        description=str(data.get("description", "")),
        raw=dict(data),
    )
