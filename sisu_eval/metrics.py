"""Per-answer scores, per-variant aggregates, bootstrap CIs and Cohen's kappa."""
from __future__ import annotations

import math
import re
from statistics import median
from typing import Any, Callable, Iterable, Mapping, Sequence

from .common import slugify, split_heading

PRODUCT_DEADLINE_S = 95.0
BOOTSTRAP_RESAMPLES = 10_000
SEED = 7


# Per answer -------------------------------------------------------------------------

def _section_slugs(section: Mapping[str, Any]) -> set[str]:
    """Slugs and explicit ids of every heading on a passage's section path."""

    parts = re.split(r"\s*(?:>|/|›|»|\|)\s*(?![^{]*\})", str(section.get("section_path") or ""))
    parts += [str(h) for h in section.get("headers") or ()]
    parts.append(str(section.get("locator") or ""))
    slugs = set()
    for part in parts:
        title, ids = split_heading(part)
        slugs |= ids
        if slugify(title):
            slugs.add(slugify(title))
    return slugs


def score_answer(question: Mapping[str, Any], row: Mapping[str, Any], judgment: Mapping[str, Any] | None
                 ) -> dict[str, Any]:
    """Numbers for one (variant, question, corpus) result. ``None`` = not applicable."""

    corpus = row["corpus"]
    answerable = question["answerable_at_snapshot"] if corpus == "snapshot" else question.get("answerable_at_head", True)
    status_key = "status_at_snapshot" if corpus == "snapshot" else "status_at_head"
    gold = [g for g in question["gold"] if g.get(status_key) == "present"] or question["gold"]
    gold_paths = {g["path"] for g in gold}
    anchors = {(g["path"], g["anchor"]) for g in gold if g.get("anchor") and g.get("anchor_verified", True)}
    presented = set(row.get("presented_docs") or ())
    sections = row.get("presented_sections") or []
    cited = {c.get("path") for c in row.get("citations") or ()}
    answered = bool(row.get("answered"))
    calls = row.get("model_calls") or []
    total_s = row.get("total_s")
    verdict = ((judgment or {}).get("correctness") or {}).get("verdict")
    support = ((judgment or {}).get("support") or {}).get("verdict")
    scored = {
        "answerable": answerable, "answered": answered,
        "evidence_doc_hit": float(bool(gold_paths & presented)) if answerable else None,
        "evidence_section_hit": (float(any(s.get("path") == path and anchor in _section_slugs(s)
                                           for path, anchor in anchors for s in sections))
                                 if answerable and anchors else None),
        "citation_gold_hit": float(bool(gold_paths & cited)) if answerable and answered else None,
        "false_refusal": float(not answered) if answerable else None,
        "correct_refusal": float(not answered) if not answerable else None,
        "correct_strict": None, "correct_lenient": None, "fully_supported": None,
        "total_s": float(total_s) if isinstance(total_s, (int, float)) else None,
        "within_product_deadline": (float(total_s <= PRODUCT_DEADLINE_S)
                                    if isinstance(total_s, (int, float)) else None),
        "model_calls": float(len(calls)),
        "output_tokens": float(sum(c.get("eval_count") or 0 for c in calls)),
        "thinking_chars": float(sum(c.get("thinking_chars") or 0 for c in calls)),
        "parse_or_call_errors": float(sum(1 for c in calls if c.get("status") not in {"ok", "partial", None})),
    }
    if answerable and verdict in {"correct", "partially_correct", "incorrect", "no_answer"}:
        scored["correct_strict"] = float(verdict == "correct")
        scored["correct_lenient"] = {"correct": 1.0, "partially_correct": 0.5}.get(verdict, 0.0)
    if answered and support in {"fully_supported", "partially_supported", "unsupported", "uncited"}:
        scored["fully_supported"] = float(support == "fully_supported")
    return scored


METRICS = (
    ("answer_rate_answerable", "answered", lambda s: s["answerable"]),
    ("correct_strict", "correct_strict", None),
    ("correct_lenient", "correct_lenient", None),
    ("evidence_doc_recall", "evidence_doc_hit", None),
    ("evidence_section_recall", "evidence_section_hit", None),
    ("citation_gold_hit", "citation_gold_hit", None),
    ("fully_supported", "fully_supported", None),
    ("false_refusal", "false_refusal", None),
    ("correct_refusal_unanswerable", "correct_refusal", None),
    ("within_95s", "within_product_deadline", None),
    ("model_calls", "model_calls", None),
    ("output_tokens", "output_tokens", None),
    ("thinking_chars", "thinking_chars", None),
    ("call_errors", "parse_or_call_errors", None),
)


# Aggregation ---------------------------------------------------------------------

def _values(scored: Sequence[Mapping[str, Any]], key: str, keep: Callable | None) -> list[float]:
    out = []
    for item in scored:
        if keep is not None and not keep(item):
            continue
        value = item.get(key)
        if value is None:
            continue
        out.append(float(value))
    return out


def _bootstrap_means(values: Sequence[float], resamples: int, seed: int):
    import numpy as np

    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=float)
    # Chunked so 10,000 x n index matrices stay small for large n.
    chunk = max(1, min(resamples, 2_000_000 // max(1, len(array))))
    for start in range(0, resamples, chunk):
        stop = min(resamples, start + chunk)
        means[start:stop] = array[rng.integers(0, len(array), size=(stop - start, len(array)))].mean(axis=1)
    return np.sort(means)


def bootstrap_ci(values: Sequence[float], *, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = SEED,
                 alpha: float = 0.05) -> tuple[float, float] | None:
    """Percentile bootstrap CI of the mean."""

    if not values:
        return None
    if len(values) == 1:
        return (float(values[0]), float(values[0]))
    means = _bootstrap_means(values, resamples, seed)
    return (float(means[int(math.floor(alpha / 2 * resamples))]),
            float(means[min(resamples - 1, int(math.ceil((1 - alpha / 2) * resamples)) - 1)]))


def paired_bootstrap(pairs: Sequence[tuple[float, float]], *, resamples: int = BOOTSTRAP_RESAMPLES,
                     seed: int = SEED) -> dict[str, Any] | None:
    """Mean of (a - b) over shared questions, 95% percentile CI, two-sided bootstrap p."""

    if not pairs:
        return None
    diffs = [a - b for a, b in pairs]
    observed = sum(diffs) / len(diffs)
    if len(diffs) == 1:
        return {"diff": observed, "ci": (observed, observed), "p_boot": 1.0, "n": 1}
    boot = _bootstrap_means(diffs, resamples, seed)
    low = float(boot[int(0.025 * resamples)])
    high = float(boot[min(resamples - 1, int(math.ceil(0.975 * resamples)) - 1)])
    at_or_below = float((boot <= 0).mean())
    at_or_above = float((boot >= 0).mean())
    return {"diff": observed, "ci": (low, high), "p_boot": min(1.0, 2 * min(at_or_below, at_or_above)), "n": len(diffs)}


def aggregate(scored: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"n": len(scored),
                              "n_answerable": sum(1 for s in scored if s["answerable"]),
                              "n_unanswerable": sum(1 for s in scored if not s["answerable"])}
    for name, key, keep in METRICS:
        values = _values(scored, key, keep)
        result[name] = {"mean": (sum(values) / len(values)) if values else None, "n": len(values),
                        "ci": bootstrap_ci(values)}
    latencies = sorted(_values(scored, "total_s", None))
    result["latency_median_s"] = median(latencies) if latencies else None
    result["latency_p90_s"] = latencies[min(len(latencies) - 1, int(math.ceil(0.9 * len(latencies))) - 1)] \
        if latencies else None
    return result


def compare(reference: Mapping[str, Mapping[str, Any]], other: Mapping[str, Mapping[str, Any]]
            ) -> dict[str, Any]:
    """Paired differences (other - reference) on keys present in both."""

    shared = sorted(set(reference) & set(other))
    out = {}
    for name, key, keep in METRICS:
        pairs = []
        for k in shared:
            a, b = other[k], reference[k]
            if keep is not None and not (keep(a) and keep(b)):
                continue
            if a.get(key) is None or b.get(key) is None:
                continue
            pairs.append((float(a[key]), float(b[key])))
        out[name] = paired_bootstrap(pairs)
    return out


# Agreement ---------------------------------------------------------------------------

def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float | None:
    pairs = [(x, y) for x, y in zip(a, b) if x and y]
    if not pairs:
        return None
    labels = sorted({x for x, _ in pairs} | {y for _, y in pairs})
    n = len(pairs)
    observed = sum(1 for x, y in pairs if x == y) / n
    expected = sum((sum(1 for x, _ in pairs if x == label) / n) * (sum(1 for _, y in pairs if y == label) / n)
                   for label in labels)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def fmt_mean_ci(entry: Mapping[str, Any] | None, *, percent: bool = True, digits: int = 1) -> str:
    if not entry or entry.get("mean") is None:
        return "n/a"
    scale = 100.0 if percent else 1.0
    mean = entry["mean"] * scale
    ci = entry.get("ci")
    if not ci:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} [{ci[0] * scale:.{digits}f}, {ci[1] * scale:.{digits}f}] (n={entry['n']})"
