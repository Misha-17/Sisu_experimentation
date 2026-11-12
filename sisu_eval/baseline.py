"""Plain RAG baseline on the same index, retrieval and authorization.

One retrieval with the original question, one generation call that cites
passages as [P#], no planning, no claim-to-passage binding, no review.
Authorization, answer budgets and resource routing come from the SISU base
class, so differences in the results come from the answering pipeline only.
"""
from __future__ import annotations

import re
import time

from sisu_reader.broker import BrokerCallError
from sisu_reader.config import thinking_allowance
from sisu_reader.engine import SisuReader
from sisu_reader.models import Answer

PROMPT_VERSION = "plain-rag-baseline-v1"
NOT_FOUND = "NOT_FOUND"
REFUSAL_TEXT = "I could not establish an answer from the inspected source passages."
SYSTEM = ("Answer the user's question using only the numbered passages. After each sentence, cite the passages "
          "that support it like [P3]. If the passages do not contain the answer, reply with exactly "
          f"{NOT_FOUND} and nothing else.")


def build_messages(question: str, evidence: list[dict]) -> list[dict]:
    blocks = []
    for row in evidence:
        heading = " > ".join(x for x in (row.get("document_title", ""), row.get("section_path", ""),
                                         row.get("locator", "")) if x)
        blocks.append(f"[{row['id']}] {heading}\n{row['text']}")
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Question: {question}\n\nPassages:\n\n" + "\n\n".join(blocks)}]


def parse_citations(text: str, evidence: list[dict]) -> list[dict]:
    by_id = {row["id"]: row for row in evidence}
    cited = []
    for alias in dict.fromkeys(re.findall(r"\[(P\d+)\]", text)):
        if alias in by_id:
            cited.append(by_id[alias])
    return cited


class PlainRag(SisuReader):
    def _ask(self, question, *, session, mode, progress, auth):
        started = time.perf_counter()
        allowed = set(self.access.allowed_document_revision_ids(auth, "document.search"))
        allowed.intersection_update(self.access.allowed_document_revision_ids(auth, "document.read"))
        allowed.intersection_update(self.access.allowed_document_revision_ids(auth, "document.cite"))
        documents = {d.document_revision_id: d for d in self.store.documents(allowed_document_revision_ids=tuple(allowed))}
        allowed = set(documents)
        calls, warnings, errors = [], [], []
        evidence = []
        text, status, cited = REFUSAL_TEXT, "not_found", []
        if documents:
            selection = self.retrieval.select({"F1": [question]}, tuple(allowed),
                                              token_budget=min(10000, self.config.context_tokens // 2))
            evidence, _ = self._validated_evidence(selection["evidence"], allowed)
            if evidence:
                output = min(3000 + thinking_allowance(self.config), self.broker.output_tokens_for("synthesis"))
                try:
                    result = self.broker.chat(role="synthesis", messages=build_messages(question, evidence),
                                              prompt_version=PROMPT_VERSION, num_predict=output,
                                              timeout_s=min(self.config.request_timeout_s, self.config.total_deadline_s),
                                              reasoning=self.config.reasoning, temperature=0.0, seed=7)
                    calls.append(result.call)
                    raw = (result.generation.content or "").strip()
                    if raw and raw != NOT_FOUND and not raw.startswith(NOT_FOUND):
                        text, cited = raw, parse_citations(raw, evidence)
                        status = "answer"
                except BrokerCallError as exc:
                    calls.append(exc.record)
                    errors.append("answer:" + type(exc).__name__)
                    status = "error"
        sources = tuple({"source_path": documents[row["document_revision_id"]].source_path,
                         "title": row.get("document_title", ""), "locator": row.get("locator", ""),
                         "quote": row["text"], "card_id": row["id"]} for row in cited)
        call_rows = []
        for call in calls:
            metrics = call.metrics or {}
            call_rows.append({"role": call.role, "status": call.status, "elapsed_s": call.elapsed_s,
                              "eval_count": metrics.get("eval_count"), "prompt_eval_count": metrics.get("prompt_eval_count"),
                              "thinking_chars": metrics.get("thinking_chars"), "think_sent": metrics.get("think_sent"),
                              "num_predict": metrics.get("num_predict"), "error": (call.error or "")[:300]})
        elapsed = time.perf_counter() - started
        return Answer(status=status, text=text, sources=sources, warnings=tuple(warnings),
                      timings={"answer_ready_s": elapsed, "total_s": elapsed},
                      debug={"pipeline": PROMPT_VERSION, "errors": errors,
                             "read_document_ids": sorted({row["document_revision_id"] for row in evidence}),
                             "evidence_cards": [{k: row.get(k) for k in ("document_revision_id", "section_path",
                                                                          "locator", "headers")} for row in evidence],
                             "model_call_rows": call_rows, "question_plan": {"facets": [{"id": "F1"}]},
                             "ablation": {"engine": "plain_rag", "reasoning": self.config.reasoning}})


__all__ = ["PlainRag", "build_messages", "parse_citations"]
