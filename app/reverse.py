"""Reverse source attribution ("which candidate sources explain the positives?").

Given reviewed positive and negative samples and a small set of candidate
source samples, find the smallest combinations of candidates that, treated as
hypothetical contamination sources under the exact same propagation rules as
:mod:`app.tracing`,

* explain every positive sample (a positive candidate explains itself with an
  empty path), and
* never reach any negative sample (a negative candidate can therefore never be
  part of a valid combination).

Every minimum-size combination is returned, each with, per positive, the batch
path from the chosen candidate source to that positive.

Algorithm
---------
With at most eight candidates the search space is tiny (``2**8`` subsets), so
exhaustive enumeration needs no approximation:

1. Run the forward closure once per *eligible* candidate (candidates that are
   themselves negative are excluded outright) to obtain its reachable samples
   and deterministic batch paths.
2. A candidate *taints* when its closure reaches a negative; such a candidate
   can never belong to a valid combination because reachability only grows when
   sources are combined.
3. Enumerate candidate subsets in increasing size; a subset is a solution when
   the union of its per-candidate positive coverage contains every positive.
4. Keep every solution of the minimum size and render the explanations.

The per-candidate closures are independent, so a path computed for a single
candidate stays valid inside any combination containing it. Output ordering is
fully deterministic: combinations sort lexicographically by their member ids
and explanations follow the request's positive order; ties for the explaining
source break on earliest arrival, then shortest path, then source id.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable

from . import tracing

#: Every candidate turned out to be a reviewed negative sample.
REASON_ALL_CANDIDATES_NEGATIVE = "all_candidate_sources_are_negative"
#: One or more positives cannot be reached from any eligible candidate.
REASON_UNEXPLAINED_POSITIVES = "unexplained_positive_samples"
#: Covering the positives is only possible via candidates whose closure
#: reaches a negative sample.
REASON_ALL_COMBOS_TAINT_NEGATIVE = "all_combinations_touch_negative"


class ReverseRequestError(Exception):
    """The request references samples absent from the snapshot."""


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def solve(
    *,
    samples: Iterable[dict[str, Any]],
    batches: Iterable[dict[str, Any]],
    candidate_sample_ids: Iterable[str],
    positive_sample_ids: Iterable[str],
    negative_sample_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Compute every minimum-size valid source combination.

    Args:
        samples: snapshot sample rows (needs ``sample_id``).
        batches: snapshot batch rows as consumed by :func:`tracing.compute`,
            including the ``samples`` rider list and live ``version``.
        candidate_sample_ids: 1-8 hypothetical source sample ids.
        positive_sample_ids: 1-12 reviewed positive sample ids.
        negative_sample_ids: up to 12 reviewed negative sample ids; samples
            absent from both lists are simply untested and impose no
            constraint.

    Returns:
        ``{"solvable": bool, "reason_code": str | None, "reason": str | None,
        "reason_details": dict | None, "combinations": [...]}``.

    Raises:
        ReverseRequestError: an unknown sample id was supplied.
    """
    candidates = _dedupe(candidate_sample_ids)
    positives = _dedupe(positive_sample_ids)
    negatives = _dedupe(negative_sample_ids)

    sample_rows = list(samples)
    batch_rows = list(batches)
    known = {row["sample_id"] for row in sample_rows}
    unknown = sorted(
        (set(candidates) | set(positives) | set(negatives)) - known
    )
    if unknown:
        raise ReverseRequestError(
            f"unknown sample(s): {', '.join(unknown)}"
        )

    pos_set = set(positives)
    neg_set = set(negatives)
    eligible = [c for c in candidates if c not in neg_set]

    if not eligible:
        return _unsolvable(
            REASON_ALL_CANDIDATES_NEGATIVE,
            "every candidate sample is a reviewed negative; a negative sample"
            " cannot be assumed to be a contamination source",
        )

    # One forward closure per eligible candidate. ``reachable[c]`` maps every
    # affected sample id to its rendered path entry (the candidate itself is
    # not in its own closure; self-explanation is handled separately).
    reachable: dict[str, dict[str, dict[str, Any]]] = {}
    for candidate in eligible:
        outcome = tracing.compute(
            samples=sample_rows,
            batches=batch_rows,
            source_sample_ids=[candidate],
        )
        reachable[candidate] = {
            row["sample_id"]: row for row in outcome["affected_samples"]
        }

    positive_coverage: dict[str, set[str]] = {}
    tainted: dict[str, set[str]] = {}
    for candidate in eligible:
        affected = set(reachable[candidate])
        positive_coverage[candidate] = (affected & pos_set) | (
            {candidate} if candidate in pos_set else set()
        )
        tainted[candidate] = affected & neg_set

    covered: set[str] = set()
    for covered_set in positive_coverage.values():
        covered |= covered_set
    missing = [p for p in positives if p not in covered]
    if missing:
        return _unsolvable(
            REASON_UNEXPLAINED_POSITIVES,
            "no eligible candidate can explain all positive samples under the"
            " 24-hour station transfer rule",
            {"unexplained_positive_sample_ids": missing},
        )

    clean = [c for c in eligible if not tainted[c]]
    minimum_combos: list[tuple[str, ...]] = []
    for size in range(1, len(clean) + 1):
        for combo in combinations(clean, size):
            combo_coverage: set[str] = set()
            for member in combo:
                combo_coverage |= positive_coverage[member]
            if pos_set <= combo_coverage:
                minimum_combos.append(combo)
        if minimum_combos:
            break

    if not minimum_combos:
        clean_coverage: set[str] = set()
        for candidate in clean:
            clean_coverage |= positive_coverage[candidate]
        blocking = sorted(
            candidate
            for candidate in eligible
            if tainted[candidate]
            and positive_coverage[candidate] - clean_coverage
        )
        return _unsolvable(
            REASON_ALL_COMBOS_TAINT_NEGATIVE,
            "every combination that covers the positives contains a candidate"
            " whose contamination reaches a reviewed negative sample",
            {"tainting_candidate_sample_ids": blocking},
        )

    rendered = [
        _render_combo(combo, positives, positive_coverage, reachable)
        for combo in sorted(minimum_combos)
    ]
    return {
        "solvable": True,
        "reason_code": None,
        "reason": None,
        "reason_details": None,
        "combinations": rendered,
    }


def _unsolvable(
    code: str, reason: str, details: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "solvable": False,
        "reason_code": code,
        "reason": reason,
        "reason_details": details,
        "combinations": [],
    }


def _render_combo(
    combo: tuple[str, ...],
    positives: list[str],
    positive_coverage: dict[str, set[str]],
    reachable: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    members = set(combo)
    explanations: list[dict[str, Any]] = []
    for positive in positives:
        if positive in members:
            # A positive candidate assumed to be a source explains itself;
            # there is no transfer path into it.
            explanations.append(
                {
                    "sample_id": positive,
                    "source_sample_id": positive,
                    "first_arrival_at": None,
                    "path": [],
                }
            )
            continue

        explainers = [
            candidate
            for candidate in combo
            if positive in positive_coverage[candidate]
        ]

        def sort_key(candidate: str) -> tuple[Any, int, str]:
            entry = reachable[candidate][positive]
            return (entry["first_arrival_at"], len(entry["path"]), candidate)

        chosen = min(explainers, key=sort_key)
        entry = reachable[chosen][positive]
        explanations.append(
            {
                "sample_id": positive,
                "source_sample_id": chosen,
                "first_arrival_at": entry["first_arrival_at"],
                "path": [dict(leg) for leg in entry["path"]],
            }
        )

    return {
        "source_sample_ids": list(combo),
        "explanations": explanations,
    }
