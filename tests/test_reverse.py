"""Reverse attribution semantics: minimum cover, negatives, paths, ties."""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from itertools import combinations

import pytest

from app import reverse, tracing
from app.reverse import (
    REASON_ALL_CANDIDATES_NEGATIVE,
    REASON_ALL_COMBOS_TAINT_NEGATIVE,
    REASON_UNEXPLAINED_POSITIVES,
)

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make_batches():
    """Two independent chains and one shared hub.

    C1 -> B1 -> B2 (+P1);  C2 -> B3 -> B4 (+P2);  B1 also reaches N via B5;
    B6 (shared-hub chain) reaches P1 too, giving alternative explanations.
    """
    return [
        {
            "batch_no": "B1", "origin": "s1", "destination": "hub",
            "departed_at": BASE.isoformat(), "arrived_at": (BASE + timedelta(hours=1)).isoformat(),
            "version": 1, "samples": ["C1", "X1"],
        },
        {
            "batch_no": "B2", "origin": "hub", "destination": "lab",
            "departed_at": (BASE + timedelta(hours=2)).isoformat(),
            "arrived_at": (BASE + timedelta(hours=3)).isoformat(),
            "version": 1, "samples": ["P1"],
        },
        {
            "batch_no": "B3", "origin": "s2", "destination": "hub2",
            "departed_at": BASE.isoformat(), "arrived_at": (BASE + timedelta(hours=1)).isoformat(),
            "version": 1, "samples": ["C2"],
        },
        {
            "batch_no": "B4", "origin": "hub2", "destination": "lab",
            "departed_at": (BASE + timedelta(hours=2)).isoformat(),
            "arrived_at": (BASE + timedelta(hours=4)).isoformat(),
            "version": 1, "samples": ["P2"],
        },
        {
            "batch_no": "B5", "origin": "hub", "destination": "store",
            "departed_at": (BASE + timedelta(hours=2)).isoformat(),
            "arrived_at": (BASE + timedelta(hours=3)).isoformat(),
            "version": 1, "samples": ["N1"],
        },
        {
            "batch_no": "B6", "origin": "s3", "destination": "hub3",
            "departed_at": BASE.isoformat(), "arrived_at": (BASE + timedelta(hours=1)).isoformat(),
            "version": 1, "samples": ["C3"],
        },
        {
            "batch_no": "B7", "origin": "hub3", "destination": "lab3",
            "departed_at": (BASE + timedelta(hours=2)).isoformat(),
            "arrived_at": (BASE + timedelta(hours=3)).isoformat(),
            "version": 1, "samples": ["P1"],
        },
    ]


def sample_rows(ids):
    return [{"sample_id": sid} for sid in ids]


ALL_IDS = ["C1", "C2", "C3", "X1", "P1", "P2", "N1"]


def test_minimum_combination_covers_all_positives():
    result = reverse.solve(
        samples=sample_rows(ALL_IDS),
        batches=make_batches(),
        candidate_sample_ids=["C1", "C2"],
        positive_sample_ids=["P1", "P2"],
    )
    assert result["solvable"] is True
    assert len(result["combinations"]) == 1
    combo = result["combinations"][0]
    assert combo["source_sample_ids"] == ["C1", "C2"]
    by_pos = {e["sample_id"]: e for e in combo["explanations"]}
    assert [leg["batch_no"] for leg in by_pos["P1"]["path"]] == ["B1", "B2"]
    assert by_pos["P1"]["source_sample_id"] == "C1"
    assert [leg["batch_no"] for leg in by_pos["P2"]["path"]] == ["B3", "B4"]
    assert by_pos["P2"]["first_arrival_at"] == BASE + timedelta(hours=4)


def test_all_minimum_combinations_are_listed():
    # C1 and C3 both reach P1 through hub: two equally sized solutions.
    result = reverse.solve(
        samples=sample_rows(ALL_IDS),
        batches=make_batches(),
        candidate_sample_ids=["C1", "C3"],
        positive_sample_ids=["P1"],
    )
    assert result["solvable"] is True
    combos = [c["source_sample_ids"] for c in result["combinations"]]
    assert combos == [["C1"], ["C3"]]


def test_negative_candidate_makes_combination_invalid():
    # C1 reaches N1; with N1 reviewed negative, only C3 may explain P1.
    result = reverse.solve(
        samples=sample_rows(ALL_IDS),
        batches=make_batches(),
        candidate_sample_ids=["C1", "C3"],
        positive_sample_ids=["P1"],
        negative_sample_ids=["N1"],
    )
    assert result["solvable"] is True
    assert [c["source_sample_ids"] for c in result["combinations"]] == [["C3"]]


def test_negative_sample_reached_is_unsolvable_with_reason():
    result = reverse.solve(
        samples=sample_rows(ALL_IDS),
        batches=make_batches(),
        candidate_sample_ids=["C1", "C2"],
        positive_sample_ids=["P1", "P2"],
        negative_sample_ids=["N1"],
    )
    assert result["solvable"] is False
    assert result["reason_code"] == REASON_ALL_COMBOS_TAINT_NEGATIVE
    assert result["combinations"] == []
    assert result["reason_details"]["tainting_candidate_sample_ids"] == ["C1"]


def test_all_candidates_negative_is_unsolvable():
    result = reverse.solve(
        samples=sample_rows(ALL_IDS),
        batches=make_batches(),
        candidate_sample_ids=["N1"],
        positive_sample_ids=["P1"],
        negative_sample_ids=["N1"],
    )
    assert result["solvable"] is False
    assert result["reason_code"] == REASON_ALL_CANDIDATES_NEGATIVE


def test_unexplained_positive_is_unsolvable_and_names_it():
    result = reverse.solve(
        samples=sample_rows(ALL_IDS + ["P9"]),
        batches=make_batches(),
        candidate_sample_ids=["C1"],
        positive_sample_ids=["P1", "P9"],
    )
    assert result["solvable"] is False
    assert result["reason_code"] == REASON_UNEXPLAINED_POSITIVES
    assert result["reason_details"]["unexplained_positive_sample_ids"] == ["P9"]


def test_positive_candidate_self_explains_with_empty_path():
    result = reverse.solve(
        samples=sample_rows(ALL_IDS),
        batches=make_batches(),
        candidate_sample_ids=["C1", "C2"],
        positive_sample_ids=["C1", "P2"],
    )
    assert result["solvable"] is True
    combo = result["combinations"][0]
    assert combo["source_sample_ids"] == ["C1", "C2"]
    self_explanation = next(
        e for e in combo["explanations"] if e["sample_id"] == "C1"
    )
    assert self_explanation["path"] == []
    assert self_explanation["first_arrival_at"] is None
    assert self_explanation["source_sample_id"] == "C1"


def test_untested_samples_are_not_negatives():
    # N1 exists but is not listed as negative: C1 remains admissible.
    result = reverse.solve(
        samples=sample_rows(ALL_IDS),
        batches=make_batches(),
        candidate_sample_ids=["C1"],
        positive_sample_ids=["P1", "N1"],
    )
    assert result["solvable"] is True
    explained = {e["sample_id"] for e in result["combinations"][0]["explanations"]}
    assert explained == {"P1", "N1"}


def test_unknown_sample_raises():
    with pytest.raises(reverse.ReverseRequestError):
        reverse.solve(
            samples=sample_rows(ALL_IDS),
            batches=make_batches(),
            candidate_sample_ids=["GHOST"],
            positive_sample_ids=["P1"],
        )


def test_window_boundary_is_identical_to_forward_engine():
    # Exactly-24h departure is reachable; one microsecond later is not.
    batches_ok = [
        {
            "batch_no": "B1", "origin": "a", "destination": "b",
            "departed_at": BASE.isoformat(),
            "arrived_at": (BASE + timedelta(hours=1)).isoformat(),
            "version": 1, "samples": ["C1"],
        },
        {
            "batch_no": "B2", "origin": "b", "destination": "c",
            "departed_at": (BASE + timedelta(hours=25)).isoformat(),
            "arrived_at": (BASE + timedelta(hours=26)).isoformat(),
            "version": 1, "samples": ["P1"],
        },
    ]
    ok = reverse.solve(
        samples=sample_rows(["C1", "P1"]),
        batches=batches_ok,
        candidate_sample_ids=["C1"],
        positive_sample_ids=["P1"],
    )
    assert ok["solvable"] is True

    batches_late = [
        dict(batches_ok[0]),
        {
            **batches_ok[1],
            "departed_at": (BASE + timedelta(hours=25, microseconds=1)).isoformat(),
        },
    ]
    late = reverse.solve(
        samples=sample_rows(["C1", "P1"]),
        batches=batches_late,
        candidate_sample_ids=["C1"],
        positive_sample_ids=["P1"],
    )
    assert late["solvable"] is False
    assert late["reason_code"] == REASON_UNEXPLAINED_POSITIVES


def test_paths_match_forward_closure():
    result = reverse.solve(
        samples=sample_rows(ALL_IDS),
        batches=make_batches(),
        candidate_sample_ids=["C1"],
        positive_sample_ids=["P1", "X1"],
    )
    forward = tracing.compute(
        samples=sample_rows(ALL_IDS),
        batches=make_batches(),
        source_sample_ids=["C1"],
    )
    forward_paths = {
        row["sample_id"]: [leg["batch_no"] for leg in row["path"]]
        for row in forward["affected_samples"]
    }
    explanations = {
        e["sample_id"]: [leg["batch_no"] for leg in e["path"]]
        for e in result["combinations"][0]["explanations"]
    }
    assert explanations["P1"] == forward_paths["P1"]
    assert explanations["X1"] == forward_paths["X1"]


# ----------------------------------------------------- brute-force cross-check


def brute_force(candidates, positives, negatives, samples, batches):
    """Enumerate every subset independently of the production algorithm."""

    def closure(source):
        outcome = tracing.compute(
            samples=samples,
            batches=batches,
            source_sample_ids=[source],
        )
        return {row["sample_id"] for row in outcome["affected_samples"]}

    eligible = [c for c in candidates if c not in set(negatives)]
    reach = {c: closure(c) for c in eligible}
    pos, neg = set(positives), set(negatives)
    solutions = []
    for size in range(1, len(eligible) + 1):
        for combo in combinations(eligible, size):
            affected: set[str] = set()
            for member in combo:
                affected |= reach[member]
            if pos <= affected | set(combo) and not (affected & neg):
                solutions.append(combo)
        if solutions:
            break
    return sorted(solutions)


def make_random_case(rng):
    sites = ["a", "b", "c", "d", "e"]
    count = rng.randint(3, 10)
    batches = []
    for i in range(count):
        origin = rng.choice(sites)
        destination = rng.choice([s for s in sites if s != origin])
        departed = BASE + timedelta(minutes=rng.randint(0, 60 * 36))
        arrived = departed + timedelta(minutes=rng.randint(10, 300))
        batches.append(
            {
                "batch_no": f"B{i:02d}",
                "origin": origin,
                "destination": destination,
                "departed_at": departed.isoformat(),
                "arrived_at": arrived.isoformat(),
                "version": 1,
                "samples": [f"S{rng.randrange(6)}"],
            }
        )
    ids = [f"S{i}" for i in range(6)]
    candidates = rng.sample(ids, rng.randint(1, min(4, len(ids))))
    rest = [s for s in ids if s not in candidates]
    rng.shuffle(rest)
    positives = rest[: rng.randint(1, max(1, len(rest)))]
    rest = [s for s in rest if s not in positives]
    negatives = rest[: rng.randint(0, len(rest))]
    return ids, batches, candidates, positives, negatives


def test_randomised_against_bruteforce():
    rng = random.Random(20261117)
    checked = 0
    for _ in range(300):
        ids, batches, candidates, positives, negatives = make_random_case(rng)
        expected = brute_force(
            candidates, positives, negatives, sample_rows(ids), batches
        )
        result = reverse.solve(
            samples=sample_rows(ids),
            batches=batches,
            candidate_sample_ids=candidates,
            positive_sample_ids=positives,
            negative_sample_ids=negatives,
        )
        if expected:
            assert result["solvable"] is True
            actual = [
                tuple(c["source_sample_ids"]) for c in result["combinations"]
            ]
            assert actual == expected
            # Every positive is explained exactly once per combination.
            for combo in result["combinations"]:
                explained = [e["sample_id"] for e in combo["explanations"]]
                assert explained == positives
                for explanation in combo["explanations"]:
                    if explanation["sample_id"] in combo["source_sample_ids"]:
                        assert explanation["path"] == []
                    else:
                        assert explanation["path"]
            checked += 1
        else:
            assert result["solvable"] is False
            assert result["reason_code"] in {
                REASON_ALL_CANDIDATES_NEGATIVE,
                REASON_UNEXPLAINED_POSITIVES,
                REASON_ALL_COMBOS_TAINT_NEGATIVE,
            }
    assert checked >= 25
