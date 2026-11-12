"""Ablation and thinking switches added for the evaluation pipeline."""
from types import SimpleNamespace

import pytest

from sisu_reader.config import Config, thinking_allowance, thinking_time_factor
from sisu_reader.ollama import OllamaClient, _think_value, _model_family

from test_general_engine import QUESTION, ScriptedBroker, clean_audit, draft, make, planned  # noqa: F401
from test_hybrid_retrieval import Encoder, Store, hr


# Thinking value per model family -------------------------------------------

def test_gemma_receives_boolean_think():
    assert _think_value("gemma4:12b", "off") is False
    assert _think_value("gemma4:12b", "low") is True
    assert _think_value("gemma4:e4b", "high") is True
    assert _model_family("gemma4:12b") == "gemma"


def test_gpt_oss_keeps_levels_and_cannot_disable():
    assert _think_value("gpt-oss:20b", "off") == "low"
    assert _think_value("gpt-oss:20b", "medium") == "medium"
    assert _model_family("gpt-oss:20b") == "gpt-oss"


def test_other_models_keep_previous_behaviour():
    assert _think_value("qwen3:14b", "off") is False
    assert _think_value("qwen3:14b", "low") == "low"
    assert _model_family("qwen3:14b") == "other"


# Budgets ---------------------------------------------------------------------

def test_allowance_and_time_factor_only_apply_when_thinking(tmp_path):
    off = Config(tmp_path, tmp_path / "w", reasoning="off", thinking_extra_tokens=2048, thinking_time_scale=3.0)
    on = Config(tmp_path, tmp_path / "w", reasoning="low", thinking_extra_tokens=2048, thinking_time_scale=3.0)
    assert thinking_allowance(off) == 0 and thinking_time_factor(off) == 1.0
    assert thinking_allowance(on) == 2048 and thinking_time_factor(on) == 3.0
    assert thinking_allowance(SimpleNamespace()) == 0


def test_ollama_ceiling_grows_with_thinking(tmp_path):
    off = Config(tmp_path, tmp_path / "w", reasoning="off")
    on = Config(tmp_path, tmp_path / "w", reasoning="low", thinking_extra_tokens=1000)
    base = OllamaClient(off, model="gemma4:12b").num_predict_ceiling
    assert OllamaClient(on, model="gemma4:12b").num_predict_ceiling == base + 1000


def test_config_refuses_thinking_with_a_deadline_that_starves_planning(tmp_path, monkeypatch):
    monkeypatch.setenv("SISU_READER_WORKSPACE", str(tmp_path / "w"))
    monkeypatch.setenv("SISU_READER_REASONING", "low")
    with pytest.raises(ValueError, match="DEADLINE_S >= 180"):
        Config.load(tmp_path)
    monkeypatch.setenv("SISU_READER_DEADLINE_S", "300")
    assert Config.load(tmp_path).thinking_allowance == 2048


def test_config_rejects_unknown_retrieval_lane(tmp_path, monkeypatch):
    monkeypatch.setenv("SISU_READER_WORKSPACE", str(tmp_path / "w"))
    monkeypatch.setenv("SISU_READER_RETRIEVAL_LANES", "sparse")
    with pytest.raises(ValueError):
        Config.load(tmp_path)
    monkeypatch.setenv("SISU_READER_RETRIEVAL_LANES", "lexical")
    monkeypatch.setenv("SISU_READER_QUESTION_PLANNING", "false")
    monkeypatch.setenv("SISU_READER_NEIGHBOR_CONTEXT", "false")
    loaded = Config.load(tmp_path)
    assert loaded.retrieval_lanes == "lexical"
    assert loaded.question_planning is False and loaded.neighbor_context is False


# Retrieval lanes and neighbour context ------------------------------------------

def _index(tmp_path, **config):
    store = Store([("a", "Weather", ["The local rain gauge records precipitation."]),
                   ("b", "Finance", ["The annual budget requires approval."])])
    return hr.HybridIndex(SimpleNamespace(workspace_dir=tmp_path, **config), store, encoder=Encoder())


def test_lexical_lane_has_no_dense_ranks(tmp_path):
    hits = _index(tmp_path, retrieval_lanes="lexical").search(["budget approval"], ["a", "b"])
    assert hits and all("dense_rank" not in ev for row in hits for ev in row["retrieval_evidence"])
    assert hits[0]["document_revision_id"] == "b"


def test_dense_lane_has_no_lexical_ranks(tmp_path):
    hits = _index(tmp_path, retrieval_lanes="dense").search(["precipitation"], ["a", "b"])
    assert hits and all("lexical_rank" not in ev for row in hits for ev in row["retrieval_evidence"])
    assert hits[0]["document_revision_id"] == "a"


def test_hybrid_lane_uses_both(tmp_path):
    hits = _index(tmp_path).search(["budget approval"], ["a", "b"])
    evidence = hits[0]["retrieval_evidence"][0]
    assert "dense_rank" in evidence and "lexical_rank" in evidence


def test_neighbor_context_switch(tmp_path):
    texts = ["Intro paragraph about the permit office.", "The chair signs every request.",
             "Closing paragraph about office hours."]
    store = Store([("a", "Permits", texts)])
    with_ctx = hr.HybridIndex(SimpleNamespace(workspace_dir=tmp_path / "1", retrieval_lanes="lexical"),
                              store, encoder=Encoder())
    without = hr.HybridIndex(SimpleNamespace(workspace_dir=tmp_path / "2", retrieval_lanes="lexical",
                                             neighbor_context=False), store, encoder=Encoder())
    rows_with = with_ctx.select({"F1": ["chair"]}, ["a"], token_budget=10000)["evidence"]
    rows_without = without.select({"F1": ["chair"]}, ["a"], token_budget=10000)["evidence"]
    assert {r["selection_reason"] for r in rows_with} == {"facet_first_pass", "adjacent_source_context"}
    assert len(rows_with) == 3
    assert [r["selection_reason"] for r in rows_without] == ["facet_first_pass"]


# Engine: planning switch and thinking budgets ---------------------------------

class ThinkingAwareBroker(ScriptedBroker):
    def output_tokens_for(self, role):
        return super().output_tokens_for(role) + thinking_allowance(self.config)


def single_facet_draft(payload):
    rows = []
    for cid, marker in (("C1", "Hold samples"), ("C2", "Release requires")):
        found = next(x for x in payload["passages"] if marker in x["text"])
        rows.append({"id": cid, "text": found["text"], "facet_ids": ["F1"], "source_ids": [found["id"]]})
    return {"claims": rows, "missing_facet_ids": []}


def test_planning_switch_skips_screen_call(make):
    engine, broker, retrieval, _ = make([single_facet_draft, clean_audit], question_planning=False)
    result = engine.ask(QUESTION)
    assert [x["role"] for x in broker.calls] == ["synthesis", "review"]
    assert result.debug["ablation"]["question_planning"] is False
    assert list(retrieval.calls[0][0]) == ["F1"]


def test_thinking_adds_tokens_to_every_call(make):
    engine, _, _, _ = make(reasoning="low", thinking_extra_tokens=1000, total_deadline_s=300.0)
    engine.broker = ThinkingAwareBroker(engine.config, [planned, draft, clean_audit])
    engine.ask(QUESTION)
    requested = {x["role"]: x["num_predict"] for x in engine.broker.calls}
    assert requested["screen"] == 640 + 1000
    assert requested["synthesis"] == 3000 + 1000
    assert requested["review"] == 1600 + 1000
    assert {x["reasoning"] for x in engine.broker.calls} == {"low"}


def test_thinking_off_keeps_product_budgets(make):
    engine, broker, _, _ = make()
    engine.ask(QUESTION)
    requested = {x["role"]: x["num_predict"] for x in broker.calls}
    assert requested == {"screen": 640, "synthesis": 3000, "review": 1600}


# Launch profiles match the evaluated variants ----------------------------------------

def test_launch_profiles_match_eval_variants(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path as _Path
    from sisu_eval.runner import load_variants, make_config

    root = _Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("run_assistant_under_test", root / "run_assistant.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    variants, _ = load_variants()
    for name, profile in runner.PROFILES.items():
        for key in [k for k in list(__import__("os").environ) if k.startswith("SISU_READER_")]:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("SISU_READER_VARIANT", name)
        runner.configure_environment()
        launched = Config.load(root)
        evaluated = make_config(variants[name], tmp_path / name, vector_cache=tmp_path / "cache")
        for field in ("model", "reasoning", "thinking_allowance", "thinking_time_factor", "context_tokens",
                      "total_deadline_s", "request_timeout_s", "max_model_calls", "max_output_tokens",
                      "inference_lock_timeout_s"):
            assert getattr(launched, field) == getattr(evaluated, field), (name, field)
    marker = (root / "VARIANT").read_text(encoding="utf-8").strip()
    assert marker in runner.PROFILES
