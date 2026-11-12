"""Mine real newcomer questions that a maintainer answered with a docs link.

Sources: GitHub Discussions (GraphQL, needs a token), GitHub issues with a
label (REST), and Discourse forums (public JSON API). Every record keeps the
thread URL so a public release can ship IDs and URLs instead of text.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Mapping

from .common import (clean_question, extract_links, html_to_text_and_links, http_json, iso, parse_time)
from .projects import Project

GITHUB_GRAPHQL = "https://api.github.com/graphql"
GITHUB_REST = "https://api.github.com"

Fetch = Callable[..., Any]

_CATEGORY_QUERY = """
query($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    discussionCategories(first: 100) { nodes { id name slug } }
  }
}"""

_DISCUSSION_QUERY = """
query($owner: String!, $name: String!, $category: ID!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    discussions(first: 40, after: $cursor, categoryId: $category,
                orderBy: {field: CREATED_AT, direction: DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number url title body createdAt authorAssociation
        author { login }
        answer { body url createdAt authorAssociation author { login } }
        comments(first: 15) {
          nodes {
            body url createdAt authorAssociation author { login }
            replies(first: 8) { nodes { body url createdAt authorAssociation author { login } } }
          }
        }
      }
    }
  }
}"""


def _matching_links(project: Project, links: Iterable[str]) -> list[str]:
    return [link for link in links if any(p.regex.match(link.split("#", 1)[0]) for p in project.link_patterns)]


def _is_maintainer(project: Project, login: str, association: str = "", *, staff: bool = False,
                   group: str = "", maintainer_groups: Iterable[str] = ()) -> bool:
    login = (login or "").casefold()
    if login and login in project.maintainers:
        return True
    if association and association.upper() in project.maintainer_associations:
        return True
    if staff:
        return True
    return bool(group) and group.casefold() in {g.casefold() for g in maintainer_groups}


def _record(project: Project, source: str, thread_id: str, *, url: str, created_at: str, title: str,
            question: str, question_links: list[str], asker: str, answer: Mapping[str, Any],
            doc_links: list[str]) -> dict[str, Any] | None:
    if len(question) < project.min_question_chars:
        return None
    if project.drop_if_question_has_gold_link:
        asked = {link.split("#", 1)[0].rstrip("/") for link in question_links}
        if any(link.split("#", 1)[0].rstrip("/") in asked for link in doc_links):
            return None
    return {"qid": f"{project.name}:{source}:{thread_id}", "project": project.name, "source": source,
            "url": url, "created_at": iso(parse_time(created_at)), "title": title.strip(),
            "question": question, "question_links": question_links, "asker": asker,
            "answer": answer["text"], "answer_url": answer["url"], "answer_author": answer["author"],
            "answer_created_at": iso(parse_time(answer["created_at"])), "answer_kind": answer["kind"],
            "doc_links": doc_links}


def _pick_answer(project: Project, asker: str, candidates: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]] | None:
    for candidate in candidates:
        if candidate["author"].casefold() == (asker or "").casefold():
            continue
        trusted = candidate["maintainer"] or (project.trust_accepted_answers and candidate["kind"] == "accepted_answer")
        if not trusted:
            continue
        links = _matching_links(project, candidate["links"])
        if links:
            return candidate, links
    return None


# GitHub Discussions --------------------------------------------------------------

def mine_github_discussions(project: Project, source: Mapping[str, Any], *, token: str, limit: int,
                            since: datetime | None = None, cache_dir: Path | None = None,
                            fetch: Fetch = http_json, log=print) -> list[dict[str, Any]]:
    if not token:
        raise ValueError("GitHub Discussions need a token: set GITHUB_TOKEN (for example `export GITHUB_TOKEN=$(gh auth token)`)")
    owner, name = source["repo"].split("/", 1)
    headers = {"Authorization": f"bearer {token}"}

    def graphql(query: str, variables: Mapping[str, Any], *, cache: bool) -> Mapping[str, Any]:
        result = fetch(source.get("graphql_url", GITHUB_GRAPHQL), method="POST",
                       payload={"query": query, "variables": dict(variables)}, headers=headers,
                       cache_dir=cache_dir if cache else None, pause_s=float(source.get("pause_s", 0.8)))
        if result.get("errors"):
            raise RuntimeError(f"GitHub GraphQL error: {result['errors']}")
        return result["data"]

    categories = graphql(_CATEGORY_QUERY, {"owner": owner, "name": name}, cache=False)[
        "repository"]["discussionCategories"]["nodes"]
    wanted = {str(x).casefold() for x in source.get("categories", ())}
    chosen = [c for c in categories if c["name"].casefold() in wanted or c["slug"].casefold() in wanted]
    if not chosen:
        names = ", ".join(sorted(c["name"] for c in categories))
        raise ValueError(f"None of {sorted(wanted)} exist in {source['repo']} discussions; available: {names}")
    missing = wanted - {c["name"].casefold() for c in chosen} - {c["slug"].casefold() for c in chosen}
    if missing:
        log(f"note: categories not found in {source['repo']} and skipped: {sorted(missing)}", file=sys.stderr)

    records: list[dict[str, Any]] = []
    seen = 0
    for category in chosen:
        cursor = None
        while len(records) < limit:
            page = graphql(_DISCUSSION_QUERY, {"owner": owner, "name": name, "category": category["id"],
                                               "cursor": cursor}, cache=True)["repository"]["discussions"]
            stop = False
            for node in page["nodes"]:
                seen += 1
                if since is not None and parse_time(node["createdAt"]) < since:
                    stop = True
                    break
                asker = (node.get("author") or {}).get("login", "")
                candidates = []
                answer = node.get("answer")
                if answer:
                    candidates.append(_github_candidate(project, answer, "accepted_answer"))
                for comment in (node.get("comments") or {}).get("nodes", []):
                    candidates.append(_github_candidate(project, comment, "maintainer_comment"))
                    for reply in (comment.get("replies") or {}).get("nodes", []):
                        candidates.append(_github_candidate(project, reply, "maintainer_reply"))
                accepted = [c for c in candidates if c["kind"] == "accepted_answer"]
                others = sorted((c for c in candidates if c["kind"] != "accepted_answer"), key=lambda c: c["created_at"])
                picked = _pick_answer(project, asker, accepted + others)
                if picked is None:
                    continue
                gold, links = picked
                record = _record(project, "github_discussions", f"{source['repo']}#{node['number']}",
                                 url=node["url"], created_at=node["createdAt"], title=node["title"],
                                 question=clean_question(node["title"], node.get("body") or ""),
                                 question_links=extract_links(node.get("body") or ""), asker=asker,
                                 answer=gold, doc_links=links)
                if record is not None:
                    records.append(record)
                    if len(records) >= limit:
                        break
            if stop or not page["pageInfo"]["hasNextPage"]:
                break
            cursor = page["pageInfo"]["endCursor"]
            log(f"{source['repo']} [{category['name']}]: scanned {seen}, kept {len(records)}", file=sys.stderr)
    return records


def _github_candidate(project: Project, node: Mapping[str, Any], kind: str) -> dict[str, Any]:
    author = (node.get("author") or {}).get("login", "")
    body = node.get("body") or ""
    return {"author": author, "text": body, "url": node.get("url", ""), "created_at": node["createdAt"],
            "kind": kind, "links": extract_links(body),
            "maintainer": _is_maintainer(project, author, node.get("authorAssociation", ""))}


# GitHub issues -------------------------------------------------------------------------

def mine_github_issues(project: Project, source: Mapping[str, Any], *, token: str, limit: int,
                       since: datetime | None = None, cache_dir: Path | None = None,
                       fetch: Fetch = http_json, log=print) -> list[dict[str, Any]]:
    repo = source["repo"]
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    pause = float(source.get("pause_s", 0.8 if token else 2.0))
    labels = ",".join(source.get("labels", ["question"]))
    records: list[dict[str, Any]] = []
    page = 1
    while len(records) < limit:
        url = (f"{GITHUB_REST}/repos/{repo}/issues?labels={labels}&state=all&per_page=100"
               f"&sort=created&direction=desc&page={page}")
        items = fetch(url, headers=headers, cache_dir=cache_dir, pause_s=pause)
        if not items:
            break
        stop = False
        for issue in items:
            if "pull_request" in issue:
                continue
            if since is not None and parse_time(issue["created_at"]) < since:
                stop = True
                break
            if not issue.get("comments"):
                continue
            comments = fetch(f"{GITHUB_REST}/repos/{repo}/issues/{issue['number']}/comments?per_page=100",
                             headers=headers, cache_dir=cache_dir, pause_s=pause)
            asker = (issue.get("user") or {}).get("login", "")
            candidates = [{"author": (c.get("user") or {}).get("login", ""), "text": c.get("body") or "",
                           "url": c.get("html_url", ""), "created_at": c["created_at"], "kind": "maintainer_comment",
                           "links": extract_links(c.get("body") or ""),
                           "maintainer": _is_maintainer(project, (c.get("user") or {}).get("login", ""),
                                                        c.get("author_association", ""))}
                          for c in comments]
            picked = _pick_answer(project, asker, candidates)
            if picked is None:
                continue
            gold, links = picked
            record = _record(project, "github_issues", f"{repo}#{issue['number']}", url=issue["html_url"],
                             created_at=issue["created_at"], title=issue["title"],
                             question=clean_question(issue["title"], issue.get("body") or ""),
                             question_links=extract_links(issue.get("body") or ""), asker=asker,
                             answer=gold, doc_links=links)
            if record is not None:
                records.append(record)
                if len(records) >= limit:
                    break
        if stop:
            break
        page += 1
        log(f"{repo} issues: page {page - 1}, kept {len(records)}", file=sys.stderr)
    return records


# Discourse -------------------------------------------------------------------------------

def mine_discourse(project: Project, source: Mapping[str, Any], *, limit: int, since: datetime | None = None,
                   cache_dir: Path | None = None, fetch: Fetch = http_json, log=print) -> list[dict[str, Any]]:
    base = source["base_url"].rstrip("/")
    pause = float(source.get("pause_s", 1.0))
    listing = fetch(f"{base}/categories.json?include_subcategories=true", cache_dir=None, pause_s=pause)
    categories = {}
    for category in listing["category_list"]["categories"]:
        categories[category["slug"].casefold()] = category
        for sub in category.get("subcategory_list") or []:
            categories[sub["slug"].casefold()] = sub
    wanted = [str(x).casefold() for x in source.get("categories", ())]
    chosen = [categories[slug] for slug in wanted if slug in categories]
    if not chosen:
        raise ValueError(f"None of {wanted} exist on {base}; available slugs: {', '.join(sorted(categories))}")
    skipped = [slug for slug in wanted if slug not in categories]
    if skipped:
        log(f"note: categories not found on {base} and skipped: {skipped}", file=sys.stderr)
    groups = source.get("maintainer_groups", ())
    records: list[dict[str, Any]] = []
    for category in chosen:
        page = 0
        while len(records) < limit:
            data = fetch(f"{base}/c/{category['slug']}/{category['id']}/l/latest.json?order=created&page={page}",
                         cache_dir=cache_dir, pause_s=pause)
            topics = data.get("topic_list", {}).get("topics", [])
            if not topics:
                break
            stop = False
            for topic in topics:
                if topic.get("pinned") or topic.get("posts_count", 0) < 2:
                    continue
                if since is not None and parse_time(topic["created_at"]) < since:
                    stop = True
                    break
                thread = fetch(f"{base}/t/{topic['id']}.json", cache_dir=cache_dir, pause_s=pause)
                posts = thread.get("post_stream", {}).get("posts", [])
                if not posts:
                    continue
                first = posts[0]
                question, question_links = html_to_text_and_links(first.get("cooked", ""))
                asker = first.get("username", "")
                candidates = []
                for post in posts[1:]:
                    text, links = html_to_text_and_links(post.get("cooked", ""))
                    candidates.append({
                        "author": post.get("username", ""), "text": text, "links": links,
                        "url": f"{base}/t/{topic['id']}/{post.get('post_number', 0)}",
                        "created_at": post["created_at"],
                        "kind": "accepted_answer" if post.get("accepted_answer") else "maintainer_post",
                        "maintainer": _is_maintainer(project, post.get("username", ""), staff=bool(
                            post.get("staff") or post.get("admin") or post.get("moderator")),
                            group=post.get("primary_group_name") or "", maintainer_groups=groups)})
                ordered = [c for c in candidates if c["kind"] == "accepted_answer"] + \
                          [c for c in candidates if c["kind"] != "accepted_answer"]
                picked = _pick_answer(project, asker, ordered)
                if picked is None:
                    continue
                gold, links = picked
                record = _record(project, "discourse", f"{base.split('//', 1)[-1]}/t/{topic['id']}",
                                 url=f"{base}/t/{topic['id']}", created_at=first["created_at"],
                                 title=thread.get("title", topic.get("title", "")),
                                 question=clean_question(thread.get("title", ""), question),
                                 question_links=question_links, asker=asker, answer=gold, doc_links=links)
                if record is not None:
                    records.append(record)
                    if len(records) >= limit:
                        break
            if stop or not data.get("topic_list", {}).get("more_topics_url"):
                break
            page += 1
            log(f"{base} [{category['slug']}]: page {page}, kept {len(records)}", file=sys.stderr)
    return records


def mine_project(project: Project, *, limit_per_source: int, token: str, since: datetime | None,
                 cache_dir: Path | None, fetch: Fetch = http_json, log=print) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source in project.sources:
        kind = source["type"]
        if kind == "github_discussions":
            found = mine_github_discussions(project, source, token=token, limit=limit_per_source, since=since,
                                            cache_dir=cache_dir, fetch=fetch, log=log)
        elif kind == "github_issues":
            found = mine_github_issues(project, source, token=token, limit=limit_per_source, since=since,
                                       cache_dir=cache_dir, fetch=fetch, log=log)
        elif kind == "discourse":
            found = mine_discourse(project, source, limit=limit_per_source, since=since, cache_dir=cache_dir,
                                   fetch=fetch, log=log)
        else:
            raise ValueError(f"Unknown source type {kind!r}")
        records.extend(found)
    unique = {}
    for record in records:
        unique.setdefault(record["qid"], record)
    return sorted(unique.values(), key=lambda r: r["created_at"])
