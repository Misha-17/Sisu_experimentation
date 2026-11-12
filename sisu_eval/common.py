"""Shared helpers: JSONL files, environment overlays, text and link handling."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Iterable, Iterator, Mapping
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
DEFAULT_DATA_DIR = REPO_ROOT / "eval_data"
USER_AGENT = "sisu-eval-miner/1.0 (research evaluation; respects rate limits)"


# Files ----------------------------------------------------------------------

def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON line") from exc
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    temporary.replace(path)
    return count


def append_jsonl(path: str | Path, row: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# Environment overlays --------------------------------------------------------

@contextmanager
def environment(overrides: Mapping[str, str], *, clear_prefix: str = "") -> Iterator[None]:
    """Temporarily set environment variables; restore exactly afterwards.

    With ``clear_prefix``, every other variable starting with that prefix is
    removed for the duration, so settings from the caller's shell cannot leak
    into a variant.
    """

    cleared = [k for k in os.environ if clear_prefix and k.startswith(clear_prefix) and k not in overrides]
    previous = {key: os.environ.get(key) for key in [*overrides, *cleared]}
    try:
        for key in cleared:
            os.environ.pop(key, None)
        for key, value in overrides.items():
            os.environ[key] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


# Time -------------------------------------------------------------------------

def parse_time(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def bucket_start(moment: datetime, granularity: str) -> datetime:
    """Start of the day/week/month containing ``moment`` (UTC)."""

    day = moment.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    if granularity == "day":
        return day
    if granularity == "week":
        return day - timedelta(days=day.weekday())
    if granularity == "month":
        return day.replace(day=1)
    raise ValueError("granularity must be day, week, or month")


def iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# Text ---------------------------------------------------------------------------

_MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\((<?)(https?://[^)\s>]+)\1?\)")
_BARE_URL = re.compile(r"https?://[^\s<>\"')\]]+")


def extract_links(text: str) -> list[str]:
    """Links from Markdown or plain text, in order, without duplicates."""

    found = [m.group(2) for m in _MARKDOWN_LINK.finditer(text or "")]
    found += [m.group(0).rstrip(".,;:!?") for m in _BARE_URL.finditer(text or "")]
    return list(dict.fromkeys(found))


def html_to_text_and_links(html: str) -> tuple[str, list[str]]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html or "", "html.parser")
    links = [a.get("href", "") for a in soup.find_all("a") if a.get("href", "").startswith(("http://", "https://"))]
    for tag in soup.find_all(["aside", "blockquote"]):
        if "quote" in (tag.get("class") or []):
            tag.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, list(dict.fromkeys(links))


def clean_question(title: str, body: str, *, max_chars: int = 2000) -> str:
    """Title plus body, code fences kept short, trimmed to ``max_chars``."""

    body = re.sub(r"```.*?```", lambda m: m.group(0)[:600], body or "", flags=re.S)
    body = re.sub(r"<!--.*?-->", "", body, flags=re.S)
    text = (title.strip() + "\n\n" + body.strip()).strip()
    return text[:max_chars]


def slugify(text: str) -> str:
    """GitHub/MkDocs/Sphinx-style heading slug."""

    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").casefold()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"[\s_-]+", "-", text).strip("-")


def heading_slugs(text: str) -> set[str]:
    """Slugs of Markdown ATX headings and reStructuredText underlined titles."""

    slugs = set()
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
        if match:
            title = re.sub(r"\{#([\w-]+)\}\s*$", "", match.group(1)).strip()
            slugs.add(slugify(title))
            explicit = re.search(r"\{#([\w-]+)\}", match.group(1))
            if explicit:
                slugs.add(explicit.group(1).casefold())
            continue
        if index + 1 < len(lines) and line.strip():
            under = lines[index + 1].strip()
            if len(under) >= max(3, len(line.strip()) - 2) and re.fullmatch(r"([=\-~^\"'`#*+])\1+", under):
                slugs.add(slugify(line.strip()))
        label = re.match(r"^\.\.\s+_([\w-]+):\s*$", line)
        if label:
            slugs.add(label.group(1).casefold())
    return slugs


# HTTP ---------------------------------------------------------------------------

class HttpError(RuntimeError):
    def __init__(self, status: int, url: str, body: str):
        super().__init__(f"HTTP {status} for {url}: {body[:300]}")
        self.status = status


def http_json(url: str, *, method: str = "GET", payload: Any = None, headers: Mapping[str, str] | None = None,
              cache_dir: Path | None = None, pause_s: float = 0.5, timeout_s: float = 60.0,
              retries: int = 4) -> Any:
    """JSON request with an optional on-disk cache and polite retries."""

    data = None if payload is None else json.dumps(payload).encode("utf-8")
    key = sha256_text(method + " " + url + " " + (data or b"").decode("utf-8"))
    cached = None if cache_dir is None else Path(cache_dir) / f"{key}.json"
    if cached is not None and cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))
    request_headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if data is not None:
        request_headers["Content-Type"] = "application/json"
    request_headers.update(headers or {})
    delay = max(pause_s, 1.0)
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, data=data, headers=request_headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                value = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            retryable = exc.code in {429, 500, 502, 503, 504} or (exc.code == 403 and "rate limit" in body.casefold())
            if not retryable or attempt == retries:
                raise HttpError(exc.code, url, body) from exc
            wait = float(exc.headers.get("Retry-After") or delay)
            time.sleep(min(wait, 900.0))
            delay *= 2
        except urllib.error.URLError:
            if attempt == retries:
                raise
            time.sleep(delay)
            delay *= 2
    if cached is not None:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(value), encoding="utf-8")
    time.sleep(pause_s)
    return value


def url_without_fragment(url: str) -> tuple[str, str]:
    parts = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, "")), parts.fragment
