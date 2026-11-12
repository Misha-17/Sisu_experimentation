"""Evaluation pipeline: mining, dated snapshots, runs, judging, metrics, ACL.

Everything runs offline: API responses are fixtures, the docs repository is a
git repo with dated commits built in a temp dir, and a scripted broker stands
in for Ollama. The SISU engine, index and access manager are the real ones.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from sisu_reader.models import GenerationResult, ModelCallRecord
from sisu_eval import acl as acl_module
from sisu_eval.common import bucket_start, clean_question, extract_links, heading_slugs, parse_time, read_jsonl
from sisu_eval.judge import judge_variants, load_judgments
from sisu_eval.labels import calibrate, calibration_export, review_export, review_import
from sisu_eval.metrics import aggregate, bootstrap_ci, cohen_kappa, compare, paired_bootstrap, score_answer
from sisu_eval.mine import mine_discourse, mine_github_discussions
from sisu_eval.projects import load_project, project_from_mapping
from sisu_eval.report import build_report, export_public
from sisu_eval.runner import load_variants, make_config, run_variant, select_variants
from sisu_eval.snapshot import build_snapshots, ensure_clone, load_questions


# Links and text ----------------------------------------------------------------

def test_fastapi_links_resolve_to_english_docs_only():
    project = load_project("fastapi")
    files = {"docs/en/docs/tutorial/body.md", "docs/en/docs/index.md", "docs/en/docs/advanced/index.md",
             "docs/es/docs/tutorial/body.md", "fastapi/routing.py"}
    assert project.resolve_link("https://fastapi.tiangolo.com/tutorial/body/#request-body", files) == \
        ("docs/en/docs/tutorial/body.md", "request-body")
    assert project.resolve_link("https://fastapi.tiangolo.com/", files) == ("docs/en/docs/index.md", "")
    assert project.resolve_link("https://fastapi.tiangolo.com/advanced/", files) == ("docs/en/docs/advanced/index.md", "")
    assert project.resolve_link("https://fastapi.tiangolo.com/es/tutorial/body/", files) is None
    assert project.resolve_link("https://github.com/fastapi/fastapi/blob/master/fastapi/routing.py", files) is None
    assert project.resolve_link("https://github.com/fastapi/fastapi/blob/master/docs/en/docs/index.md", files) == \
        ("docs/en/docs/index.md", "")


def test_pydantic_versioned_and_devguide_rst_links():
    pydantic = load_project("pydantic")
    files = {"docs/concepts/models.md", "docs/usage/models.md"}
    assert pydantic.resolve_link("https://docs.pydantic.dev/latest/concepts/models/#nested", files) == \
        ("docs/concepts/models.md", "nested")
    assert pydantic.resolve_link("https://docs.pydantic.dev/2.7/concepts/models/", files)[0] == "docs/concepts/models.md"
    assert pydantic.resolve_link("https://docs.pydantic.dev/usage/models/", files)[0] == "docs/usage/models.md"
    devguide = load_project("cpython-devguide")
    files = {"getting-started/setup-building.rst", "index.rst", "_static/x.css"}
    assert devguide.resolve_link("https://devguide.python.org/getting-started/setup-building/", files) == \
        ("getting-started/setup-building.rst", "")
    assert devguide.resolve_link("https://devguide.python.org/", files) == ("index.rst", "")
    assert devguide.indexed_name("index.rst") == "index.rst.txt"


def test_text_helpers():
    assert extract_links("See [docs](https://a.dev/x/) and https://b.dev/y.") == ["https://a.dev/x/", "https://b.dev/y"]
    md = "# Request Body\n\nText\n\n## Use `Body` {#custom-id}\n"
    assert {"request-body", "custom-id"} <= heading_slugs(md)
    rst = ".. _setup:\n\nSetup and building\n==================\n"
    assert {"setup", "setup-and-building"} <= heading_slugs(rst)
    assert clean_question("Title", "body " * 1000).startswith("Title\n\nbody")
    assert bucket_start(parse_time("2024-03-14T10:00:00Z"), "week").isoformat().startswith("2024-03-11")
    assert bucket_start(parse_time("2024-03-14T10:00:00Z"), "month").isoformat().startswith("2024-03-01")


# Mining ------------------------------------------------------------------------------

DOCS = "https://fastapi.tiangolo.com/tutorial/body/"


def _discussion(number, created, *, answer=None, comments=(), body="How do I declare a request body for my endpoint please?",
                asker="newcomer"):
    return {"number": number, "url": f"https://github.com/fastapi/fastapi/discussions/{number}", "title": f"Q{number}",
            "body": body, "createdAt": created, "authorAssociation": "NONE", "author": {"login": asker},
            "answer": answer, "comments": {"nodes": list(comments)}}


def _comment(body, created, association="MEMBER", login="maintainer", replies=()):
    return {"body": body, "url": "https://github.com/c", "createdAt": created, "authorAssociation": association,
            "author": {"login": login}, "replies": {"nodes": list(replies)}}


def test_github_discussions_picks_maintainer_docs_answers():
    project = load_project("fastapi")
    nodes = [
        _discussion(1, "2024-05-01T00:00:00Z", answer=_comment(f"Read {DOCS}", "2024-05-02T00:00:00Z")),
        _discussion(2, "2024-04-01T00:00:00Z", comments=[_comment(f"see {DOCS}", "2024-04-02T00:00:00Z",
                                                                  association="NONE", login="random")]),
        _discussion(3, "2024-03-01T00:00:00Z", comments=[_comment(f"I solved it: {DOCS}", "2024-03-02T00:00:00Z",
                                                                  association="MEMBER", login="newcomer")]),
        _discussion(4, "2024-02-01T00:00:00Z", body=f"I read {DOCS} but still confused about bodies",
                    comments=[_comment(f"Again: {DOCS}", "2024-02-02T00:00:00Z")]),
        _discussion(5, "2024-01-01T00:00:00Z", comments=[_comment("no link", "2024-01-02T00:00:00Z",
                     replies=[_comment(f"[body docs]({DOCS}#request-body)", "2024-01-03T00:00:00Z", association="OWNER")])]),
        _discussion(6, "2023-01-01T00:00:00Z", answer=_comment(f"Read {DOCS}", "2023-01-02T00:00:00Z")),
    ]

    def fetch(url, **kwargs):
        query = kwargs["payload"]["query"]
        if "discussionCategories" in query:
            return {"data": {"repository": {"discussionCategories": {"nodes": [
                {"id": "C1", "name": "Questions", "slug": "questions"}, {"id": "C2", "name": "Ideas", "slug": "ideas"}]}}}}
        assert kwargs["payload"]["variables"]["category"] == "C1"
        return {"data": {"repository": {"discussions": {"pageInfo": {"hasNextPage": False, "endCursor": None},
                                                        "nodes": nodes}}}}

    records = mine_github_discussions(project, project.sources[0], token="t", limit=10,
                                      since=parse_time("2023-06-01T00:00:00Z"), fetch=fetch, log=lambda *a, **k: None)
    assert [r["qid"].rsplit("#", 1)[1] for r in records] == ["1", "5"]
    assert records[0]["answer_kind"] == "accepted_answer"
    assert records[1]["answer_kind"] == "maintainer_reply" and records[1]["doc_links"] == [DOCS + "#request-body"]
    with pytest.raises(ValueError, match="GITHUB_TOKEN"):
        mine_github_discussions(project, project.sources[0], token="", limit=1, fetch=fetch)


def test_discourse_accepted_answer_is_trusted_when_configured():
    devguide = load_project("cpython-devguide")
    link = "https://devguide.python.org/getting-started/setup-building/"

    def fetch(url, **kwargs):
        if url.endswith("categories.json?include_subcategories=true"):
            return {"category_list": {"categories": [{"id": 23, "slug": "core-dev", "subcategory_list": []}]}}
        if "/l/latest.json" in url:
            return {"topic_list": {"topics": [{"id": 9, "title": "Build fails", "created_at": "2024-02-01T00:00:00Z",
                                               "posts_count": 2}], "more_topics_url": None}}
        return {"title": "Build fails on macOS", "post_stream": {"posts": [
            {"username": "new", "cooked": "<p>My build of CPython fails on macOS with an SSL error, what do I do?</p>",
             "created_at": "2024-02-01T00:00:00Z", "post_number": 1},
            {"username": "helper", "cooked": f'<p>Follow <a href="{link}">the setup guide</a>.</p>',
             "created_at": "2024-02-01T05:00:00Z", "post_number": 2, "accepted_answer": True}]}}

    records = mine_discourse(devguide, devguide.sources[0], limit=5, fetch=fetch, log=lambda *a, **k: None)
    assert len(records) == 1 and records[0]["answer_kind"] == "accepted_answer"
    assert records[0]["doc_links"] == [link]


# Dated snapshots ------------------------------------------------------------------------

def _git(repo, *args, date=None):
    env = dict(os.environ)
    if date:
        env.update(GIT_AUTHOR_DATE=date, GIT_COMMITTER_DATE=date)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
                   check=True, capture_output=True, env=env)


@pytest.fixture
def docs_repo(tmp_path):
    repo = tmp_path / "upstream"
    (repo / "docs").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    (repo / "docs" / "install.md").write_text(
        "# Installing\n\nInstall the toolkit with pip install toolkit. The install needs Python 3.11.\n", encoding="utf-8")
    (repo / "docs" / "notes.rst").write_text("Release notes\n=============\n\nNothing yet.\n", encoding="utf-8")
    (repo / "src.py").write_text("print(1)\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "install docs", date="2024-01-10T12:00:00Z")
    (repo / "docs" / "deploy.md").write_text(
        "# Deploying\n\n## Docker\n\nDeploy the toolkit with the official docker image toolkit/server.\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "deploy docs", date="2024-03-20T12:00:00Z")
    return repo


def _project(repo):
    return project_from_mapping({
        "name": "toy", "docs_repo": str(repo), "docs_roots": ["docs"],
        "link_patterns": [{"regex": "^https://toy\\.dev/(?P<path>[^?#]*)", "root": "docs"}],
        "sources": [{"type": "github_discussions", "repo": "toy/toy", "categories": ["Q&A"]}]})


def _mined():
    base = {"project": "toy", "source": "github_discussions", "question_links": [], "asker": "a",
            "answer_author": "m", "answer_kind": "accepted_answer", "title": "t"}
    return [
        {**base, "qid": "toy:1", "url": "https://x/1", "created_at": "2024-02-14T10:00:00Z",
         "question": "How do I install the toolkit with pip?", "answer": "Use pip install toolkit.",
         "answer_url": "https://x/1#a", "answer_created_at": "2024-02-14T11:00:00Z",
         "doc_links": ["https://toy.dev/install/#installing"]},
        {**base, "qid": "toy:2", "url": "https://x/2", "created_at": "2024-02-15T10:00:00Z",
         "question": "How do I deploy the toolkit with docker?", "answer": "Use the docker image toolkit/server.",
         "answer_url": "https://x/2#a", "answer_created_at": "2024-04-01T11:00:00Z",
         "doc_links": ["https://toy.dev/deploy/#docker"]},
        {**base, "qid": "toy:3", "url": "https://x/3", "created_at": "2024-02-16T10:00:00Z",
         "question": "Where is the source of the printer?", "answer": "See src.py",
         "answer_url": "https://x/3#a", "answer_created_at": "2024-02-16T11:00:00Z",
         "doc_links": ["https://toy.dev/missing/"]},
    ]


@pytest.fixture
def snapshotted(tmp_path, docs_repo):
    project = _project(docs_repo)
    data = tmp_path / "data"
    ensure_clone(project, data)
    summary = build_snapshots(project, _mined(), data, granularity="week")
    return project, data, summary


def test_snapshots_use_question_date_and_detect_docs_added_later(snapshotted):
    project, data, summary = snapshotted
    questions = {q["qid"]: q for q in load_questions(project, data)}
    assert set(questions) == {"toy:1", "toy:2"}
    assert summary["drop_reasons"] == {"links_not_resolvable_to_docs_files": 1}
    q1, q2 = questions["toy:1"], questions["toy:2"]
    assert q1["answerable_at_snapshot"] and q1["gold"][0]["path"] == "docs/install.md"
    assert q1["gold"][0]["anchor"] == "installing"
    assert not q2["answerable_at_snapshot"] and q2["answerable_at_head"]
    assert q2["gold"][0]["status_at_snapshot"] == "file_added_later"
    assert q1["snapshot_commit"] == q2["snapshot_commit"] != q2["head_commit"]
    tree = data / "toy" / "snapshots" / q1["snapshot_commit"] / "tree"
    assert (tree / "docs" / "install.md").exists() and not (tree / "docs" / "deploy.md").exists()
    assert (tree / "docs" / "notes.rst.txt").exists() and not (tree / "src.py").exists()
    head_tree = data / "toy" / "snapshots" / q2["head_commit"] / "tree"
    assert (head_tree / "docs" / "deploy.md").exists()


# Variants ------------------------------------------------------------------------------

def test_every_variant_loads_a_valid_config(tmp_path):
    variants, reference = load_variants()
    assert reference == "normal"
    assert {"normal", "thinking", "normal_long_budget", "plain_rag"} <= set(variants)
    for variant in variants.values():
        config = make_config(variant, tmp_path / variant.name, vector_cache=tmp_path / "cache")
        assert config.model.startswith("gemma4:")
    thinking = make_config(variants["thinking"], tmp_path / "t", vector_cache=tmp_path / "c")
    normal = make_config(variants["normal"], tmp_path / "n", vector_cache=tmp_path / "c")
    assert thinking.thinking_allowance == 3072 and thinking.thinking_time_factor == 4.0
    assert thinking.total_deadline_s == 420 and thinking.max_output_tokens == 65536
    assert normal.thinking_allowance == 0 and normal.total_deadline_s == 95
    with pytest.raises(ValueError):
        select_variants(variants, "normal,unknown")


# End-to-end with a scripted model -----------------------------------------------------------

class GenericBroker:
    """Answers by role: one facet, cite the first passage, clean review."""

    def __init__(self, config):
        from sisu_reader.config import thinking_allowance
        self.config = config
        self.calls = []
        self.current_answer_budget = None
        self._allowance = thinking_allowance(config)

    def answer_budget(self, config):
        from contextlib import contextmanager
        from sisu_reader.answer_budget import AnswerBudget
        from sisu_reader.grounded_answer import AUDIT_PROMPT_VERSION

        @contextmanager
        def scope():
            self.current_answer_budget = AnswerBudget(config, repair_prompt_version=AUDIT_PROMPT_VERSION)
            try:
                yield self.current_answer_budget
            finally:
                self.current_answer_budget.close()
                self.current_answer_budget = None
        return scope()

    def output_tokens_for(self, role):
        return getattr(self.config, role + "_output_tokens") + self._allowance

    def chat(self, **kwargs):
        role = kwargs["role"]
        admission = self.current_answer_budget.admit(role=role, prompt_version=kwargs["prompt_version"],
                                                    requested_tokens=kwargs["num_predict"])
        self.calls.append(kwargs)
        if kwargs["prompt_version"].startswith("plain-rag"):
            raw = "Use the documented command. [P1]"
        else:
            payload = json.loads(kwargs["messages"][1]["content"])
            if role == "screen":
                raw = json.dumps({"facets": [{"id": "F1", "question": payload.get("question", "q"), "queries": []}]})
            elif role == "synthesis":
                first = payload["passages"][0]
                raw = json.dumps({"claims": [{"id": "C1", "text": first["text"][:200], "facet_ids": ["F1"],
                                              "source_ids": [first["id"]]}], "missing_facet_ids": []})
            else:
                raw = json.dumps({"unsupported_claim_ids": [], "missing_facet_ids": [], "searches": [], "reason": "ok"})
        metrics = {"eval_count": 40, "prompt_eval_count": 400, "thinking_chars": 0, "think_sent": "false"}
        self.current_answer_budget.settle(admission, metrics=metrics)
        record = ModelCallRecord(str(len(self.calls)), role, "scripted", kwargs["prompt_version"], 0.01, "ok",
                                 metrics=metrics, raw_output=raw, prompt="")
        return SimpleNamespace(generation=GenerationResult(raw, metrics=metrics), call=record)


def scripted_factory(variant, config, *, vector_cache):
    from sisu_eval.common import environment
    with environment({"SISU_READER_VECTOR_CACHE_DIR": str(vector_cache)}):
        if variant.engine == "plain_rag":
            from sisu_eval.baseline import PlainRag as engine_class
        else:
            from sisu_reader.engine import SisuReader as engine_class
        engine = engine_class(config, broker=GenericBroker(config))
    engine.store.load()
    engine._loaded = True
    engine.retrieval.ensure()
    return engine


class FakeJudge:
    def chat(self, *, messages, format, **kwargs):
        if "verdict" in format["properties"] and "unsupported_statements" in format["properties"]:
            return GenerationResult(json.dumps({"verdict": "fully_supported", "unsupported_statements": [],
                                                "rationale": "quoted"}))
        payload = json.loads(messages[1]["content"])
        good = "toolkit" in payload["candidate_answer"].casefold() or "documented" in payload["candidate_answer"]
        return GenerationResult(json.dumps({"verdict": "correct" if good else "incorrect", "rationale": "r"}))


@pytest.fixture
def ran(snapshotted):
    project, data, _ = snapshotted
    variants, reference = load_variants()
    questions = load_questions(project, data)
    chosen = select_variants(variants, "normal,thinking,plain_rag,no_planning")
    for variant in chosen:
        summary = run_variant(data, "toy", variant, questions, head_check=True, engine_factory=scripted_factory,
                              log=lambda *a, **k: None)
        assert summary["errors"] == 0, read_jsonl(Path(summary["output"]))
    return project, data, variants, reference, chosen, questions


def test_runs_record_citations_presented_docs_and_calls(ran):
    project, data, variants, *_ = ran
    rows = {(r["qid"], r["corpus"]): r for r in read_jsonl(data / "toy" / "runs" / "normal.jsonl")}
    assert set(rows) == {("toy:1", "snapshot"), ("toy:2", "snapshot"), ("toy:2", "head")}
    first = rows[("toy:1", "snapshot")]
    assert first["answered"] and first["citations"][0]["path"].startswith("docs/")
    assert "docs/install.md" in first["presented_docs"]
    assert [c["role"] for c in first["model_calls"]] == ["screen", "synthesis", "review"]
    assert all("docs/deploy.md" not in r["presented_docs"] for k, r in rows.items() if k[1] == "snapshot")
    assert "docs/deploy.md" in rows[("toy:2", "head")]["presented_docs"]
    thinking = read_jsonl(data / "toy" / "runs" / "thinking.jsonl")[0]
    assert thinking["ablation"]["reasoning"] == "high" and thinking["ablation"]["thinking_extra_tokens"] == 3072
    no_plan = read_jsonl(data / "toy" / "runs" / "no_planning.jsonl")[0]
    assert [c["role"] for c in no_plan["model_calls"]] == ["synthesis", "review"]
    baseline = read_jsonl(data / "toy" / "runs" / "plain_rag.jsonl")[0]
    assert baseline["answered"] and [c["role"] for c in baseline["model_calls"]] == ["synthesis"]
    assert baseline["citations"] and baseline["presented_sections"]
    # Resume: a second run adds nothing.
    again = run_variant(data, "toy", variants["normal"], load_questions(project, data), head_check=True,
                        engine_factory=scripted_factory, log=lambda *a, **k: None)
    assert again["jobs"] == 0


def test_judge_report_calibration_and_public_export(ran):
    project, data, variants, reference, chosen, questions = ran
    names = [v.name for v in chosen]
    counts = judge_variants(data, "toy", names, {q["qid"]: q for q in questions}, judge_model="fake-judge",
                            client=FakeJudge(), log=lambda *a, **k: None)
    assert counts["judged"] == 12
    assert judge_variants(data, "toy", names, {q["qid"]: q for q in questions}, judge_model="fake-judge",
                          client=FakeJudge(), log=lambda *a, **k: None)["judged"] == 0
    judgments = load_judgments(data, "toy", "fake-judge")
    assert judgments[("normal", "toy:1", "snapshot")]["support"]["verdict"] == "fully_supported"
    report = build_report(data, "toy", chosen, reference, "fake-judge", project=project)
    normal = report["variants"]["normal"]["snapshot"]
    assert normal["n"] == 2 and normal["n_answerable"] == 1
    assert normal["evidence_doc_recall"]["mean"] == 1.0
    assert report["variants"]["normal"]["head_check"]["n"] == 1
    assert "vs_reference" in report["variants"]["thinking"]
    markdown = (data / "toy" / "report.md").read_text(encoding="utf-8")
    assert "Judge not yet checked" in markdown and "| thinking |" in markdown
    exported = calibration_export(data, "toy", "fake-judge", names, n=4)
    sheet = Path(exported["sheet"])
    rows = list(csv.DictReader(sheet.open(encoding="utf-8-sig")))
    assert len(rows) == 4 and "judge" not in ",".join(rows[0].keys())
    for row in rows:
        row.update(rater_1_correctness="correct", rater_2_correctness="correct",
                   rater_1_support="fully_supported", rater_2_support="fully_supported")
    with sheet.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = calibrate(data, "toy", sheet)
    assert result["correctness"]["n_rated_by_both"] == 4
    public = read_jsonl(export_public(data, "toy"))
    assert public and "question" not in public[0] and "answer" not in public[0]


def test_review_round_trip_drops_and_overrides(snapshotted, tmp_path):
    project, data, _ = snapshotted
    out = tmp_path / "review.csv"
    assert review_export(data, "toy", out) == 2
    rows = list(csv.DictReader(out.open(encoding="utf-8-sig")))
    rows[0]["keep"] = "n"
    rows[1]["answerable_override"] = "y"
    with out.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = review_import(data, "toy", out)
    assert summary["dropped"] == 1 and summary["overrides"] == 1
    from sisu_eval.labels import load_labels
    kept = load_questions(project, data, labels=load_labels(data, "toy"))
    assert [q["qid"] for q in kept] == ["toy:2"] and kept[0]["answerable_at_snapshot"] is True


def test_acl_user_never_sees_forbidden_half(snapshotted, monkeypatch):
    project, data, _ = snapshotted
    variants, _ = load_variants()
    questions = load_questions(project, data)
    # Put install.md in the allowed half and deploy.md in the forbidden half.
    monkeypatch.setattr(acl_module, "in_allowed_half", lambda path: path == "docs/install.md")
    summary = acl_module.run_acl(data, "toy", variants["normal"], questions, engine_factory=scripted_factory,
                                 log=lambda *a, **k: None)
    assert summary["questions"] == 2
    assert summary["answers_with_leaked_citations"] == 0
    assert summary["answers_with_leaked_presented_passages"] == 0
    rows = {r["qid"]: r for r in read_jsonl(data / "toy" / "acl" / "normal.jsonl")}
    assert rows["toy:1"]["gold_allowed"] and rows["toy:1"]["answered"]
    assert not rows["toy:2"]["gold_allowed"]
    assert all(p == "docs/install.md" for p in rows["toy:2"].get("cited_paths", []))


# Metrics -----------------------------------------------------------------------------

def test_bootstrap_and_kappa():
    assert bootstrap_ci([1.0] * 20) == (1.0, 1.0)
    low, high = bootstrap_ci([0.0, 1.0] * 50)
    assert 0.38 < low < 0.5 < high < 0.62
    assert bootstrap_ci([0.0, 1.0] * 50) == (low, high)
    paired = paired_bootstrap([(1.0, 0.0)] * 30)
    assert paired["diff"] == 1.0 and paired["p_boot"] == 0.0
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == 0.0
    assert cohen_kappa(["a", ""], ["a", "b"]) == 1.0


def test_score_answer_rules():
    question = {"answerable_at_snapshot": True, "answerable_at_head": True,
                "gold": [{"path": "docs/a.md", "anchor": "setup", "status_at_snapshot": "present",
                          "status_at_head": "present"}]}
    row = {"corpus": "snapshot", "answered": True, "presented_docs": ["docs/a.md"],
           "presented_sections": [{"path": "docs/a.md", "section_path": "Guide > Setup", "headers": []}],
           "citations": [{"path": "docs/b.md"}], "model_calls": [{"eval_count": 10, "status": "ok"}], "total_s": 100.0}
    judged = {"correctness": {"verdict": "partially_correct"}, "support": {"verdict": "unsupported"}}
    scored = score_answer(question, row, judged)
    assert scored["evidence_doc_hit"] == 1.0 and scored["evidence_section_hit"] == 1.0
    assert scored["citation_gold_hit"] == 0.0 and scored["correct_lenient"] == 0.5 and scored["correct_strict"] == 0.0
    assert scored["fully_supported"] == 0.0 and scored["within_product_deadline"] == 0.0
    refused = score_answer({**question, "answerable_at_snapshot": False}, {**row, "answered": False}, None)
    assert refused["correct_refusal"] == 1.0 and refused["evidence_doc_hit"] is None
    agg = aggregate([scored, refused])
    assert agg["n"] == 2 and agg["n_answerable"] == 1
    diff = compare({("q", "snapshot"): scored}, {("q", "snapshot"): {**scored, "correct_strict": 1.0}})
    assert diff["correct_strict"]["diff"] == 1.0


def test_cli_help_runs():
    result = subprocess.run([os.sys.executable, "-m", "sisu_eval", "--help"], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parents[1])
    assert result.returncode == 0 and "mine" in result.stdout and "calibrate" in result.stdout


def test_shell_settings_do_not_leak_into_variants(tmp_path, monkeypatch):
    monkeypatch.setenv("SISU_READER_CLAIM_REVIEW", "false")
    monkeypatch.setenv("SISU_READER_RETRIEVAL_LANES", "dense")
    variants, _ = load_variants()
    config = make_config(variants["normal"], tmp_path / "w", vector_cache=tmp_path / "c")
    assert config.claim_review is True and config.retrieval_lanes == "hybrid"
    assert os.environ["SISU_READER_CLAIM_REVIEW"] == "false"


def test_fastapi_exports_code_examples_as_text():
    project = load_project("fastapi")
    assert project.is_doc_path("docs_src/body/tutorial001.py")
    assert project.indexed_name("docs_src/body/tutorial001.py") == "docs_src/body/tutorial001.py.txt"
    assert not project.is_doc_path("fastapi/routing.py")


def test_questions_from_before_the_docs_layout_are_dropped_not_fatal(tmp_path):
    repo = tmp_path / "moved"
    (repo / "old").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    (repo / "old" / "install.md").write_text("# Installing\n\nOld layout.\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "old layout", date="2023-11-10T12:00:00Z")
    (repo / "docs").mkdir()
    (repo / "old" / "install.md").rename(repo / "docs" / "install.md")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "move docs", date="2024-01-10T12:00:00Z")
    project = _project(repo)
    data = tmp_path / "data"
    ensure_clone(project, data)
    early = {**_mined()[0], "qid": "toy:early", "created_at": "2023-12-14T10:00:00Z",
             "answer_created_at": "2024-01-20T11:00:00Z"}
    late = {**_mined()[0], "qid": "toy:late", "created_at": "2024-02-14T10:00:00Z",
            "answer_created_at": "2024-02-14T11:00:00Z"}
    summary = build_snapshots(project, [early, late], data)
    assert summary["questions_kept"] == 1
    assert summary["drop_reasons"] == {"docs_layout_differs_at_question_date": 1}
    assert summary["kept_by_year"] == {"2024": 1} and summary["dropped_by_year"] == {"2023": 1}
    assert summary["granularity"] == "month"
