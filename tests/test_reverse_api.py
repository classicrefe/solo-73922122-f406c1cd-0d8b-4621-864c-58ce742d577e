"""Reverse attribution API: routes, validation, snapshots, failure isolation."""
from __future__ import annotations

import sqlite3
import threading

from fastapi.testclient import TestClient

from conftest import batch, ingest

from app import reverse as reverse_module
from app.main import create_app
from app.store import Store


def seed(client):
    ingest(
        client,
        sources=["C1"],
        samples=[
            {"sample_id": "C1", "batch_nos": ["B1"]},
            {"sample_id": "C2", "batch_nos": ["B3"]},
            {"sample_id": "P1", "batch_nos": ["B2"]},
            {"sample_id": "P2", "batch_nos": ["B4"]},
            {"sample_id": "N1", "batch_nos": ["B5"]},
            {"sample_id": "U1", "batch_nos": []},
        ],
        batches=[
            batch("B1", "s1", "hub", "2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"),
            batch("B2", "hub", "lab", "2026-01-01T02:00:00Z", "2026-01-01T03:00:00Z"),
            batch("B3", "s2", "hub2", "2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"),
            batch("B4", "hub2", "lab", "2026-01-01T02:00:00Z", "2026-01-01T04:00:00Z"),
            batch("B5", "hub", "store", "2026-01-01T02:00:00Z", "2026-01-01T03:30:00Z"),
        ],
    )


def reverse(client, **payload):
    return client.post("/reverse-traces", json=payload)


def test_reverse_route_in_openapi(client):
    schema = client.get("/openapi.json").json()
    assert "/reverse-traces" in schema["paths"]
    docs = client.get("/docs")
    assert docs.status_code == 200


def test_reverse_requires_dataset(client):
    response = reverse(
        client,
        candidate_sample_ids=["C1"],
        positive_sample_ids=["P1"],
    )
    assert response.status_code == 404


def test_solvable_response_has_minimum_combo_and_paths(client):
    seed(client)
    response = reverse(
        client,
        candidate_sample_ids=["C1", "C2"],
        positive_sample_ids=["P1", "P2"],
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["status"] == "complete"
    assert body["solvable"] is True
    assert body["combination_count"] == 1
    assert body["minimum_source_count"] == 2
    combo = body["combinations"][0]
    assert combo["source_sample_ids"] == ["C1", "C2"]
    by_pos = {e["sample_id"]: e for e in combo["explanations"]}
    assert [leg["batch_no"] for leg in by_pos["P1"]["path"]] == ["B1", "B2"]
    assert [leg["version"] for leg in by_pos["P1"]["path"]] == [1, 1]
    assert by_pos["P2"]["first_arrival_at"] == "2026-01-01T04:00:00Z"
    # Snapshot header is pinned like the forward trace.
    assert body["snapshot"]["batch_versions"] == {
        "B1": 1, "B2": 1, "B3": 1, "B4": 1, "B5": 1,
    }
    assert {b["batch_no"] for b in body["snapshot"]["batches"]} == {
        "B1", "B2", "B3", "B4", "B5",
    }


def test_self_explanation_serialises_empty_path(client):
    seed(client)
    body = reverse(
        client,
        candidate_sample_ids=["C1", "C2"],
        positive_sample_ids=["C1", "P2"],
    ).json()
    explanation = next(
        e
        for e in body["combinations"][0]["explanations"]
        if e["sample_id"] == "C1"
    )
    assert explanation == {
        "sample_id": "C1",
        "source_sample_id": "C1",
        "first_arrival_at": None,
        "path": [],
    }


def test_negative_constraint_filters_combinations(client):
    seed(client)
    # Without negatives, C1 alone explains P1. With N1 negative it is barred.
    blocked = reverse(
        client,
        candidate_sample_ids=["C1"],
        positive_sample_ids=["P1"],
        negative_sample_ids=["N1"],
    ).json()
    assert blocked["solvable"] is False
    assert blocked["reason_code"] == "all_combinations_touch_negative"
    assert blocked["combinations"] == []
    assert blocked["combination_count"] == 0
    assert blocked["minimum_source_count"] is None

    # An untested sample does not act as a negative.
    allowed = reverse(
        client,
        candidate_sample_ids=["C1"],
        positive_sample_ids=["P1"],
    ).json()
    assert allowed["solvable"] is True


def test_negative_candidate_is_rejected_from_combos(client):
    seed(client)
    body = reverse(
        client,
        candidate_sample_ids=["N1", "C2"],
        positive_sample_ids=["P2"],
        negative_sample_ids=["N1"],
    ).json()
    assert body["solvable"] is True
    assert body["combinations"][0]["source_sample_ids"] == ["C2"]


def test_all_candidates_negative(client):
    seed(client)
    body = reverse(
        client,
        candidate_sample_ids=["N1"],
        positive_sample_ids=["P1"],
        negative_sample_ids=["N1"],
    ).json()
    assert body["solvable"] is False
    assert body["reason_code"] == "all_candidate_sources_are_negative"


def test_unexplained_positive_names_the_sample(client):
    seed(client)
    body = reverse(
        client,
        candidate_sample_ids=["C2"],
        positive_sample_ids=["P1", "P2"],
    ).json()
    assert body["solvable"] is False
    assert body["reason_code"] == "unexplained_positive_samples"
    assert body["reason_details"] == {
        "unexplained_positive_sample_ids": ["P1"]
    }


def test_unknown_sample_is_404(client):
    seed(client)
    response = reverse(
        client,
        candidate_sample_ids=["GHOST"],
        positive_sample_ids=["P1"],
    )
    assert response.status_code == 404
    assert "unknown sample" in response.json()["message"]


def test_payload_validation(client):
    seed(client)
    # Too many candidates.
    too_many = reverse(
        client,
        candidate_sample_ids=[f"C{i}" for i in range(9)],
        positive_sample_ids=["P1"],
    )
    assert too_many.status_code == 422
    assert too_many.json()["code"] == "validation_error"

    # Empty positives.
    empty = reverse(
        client,
        candidate_sample_ids=["C1"],
        positive_sample_ids=[],
    )
    assert empty.status_code == 422

    # Duplicates.
    dup = reverse(
        client,
        candidate_sample_ids=["C1", "C1"],
        positive_sample_ids=["P1"],
    )
    assert dup.status_code == 422

    # Positive and negative overlap.
    overlap = reverse(
        client,
        candidate_sample_ids=["C1"],
        positive_sample_ids=["P1"],
        negative_sample_ids=["P1"],
    )
    assert overlap.status_code == 422

    # Unknown field is forbidden.
    extra = reverse(
        client,
        candidate_sample_ids=["C1"],
        positive_sample_ids=["P1"],
        bogus=1,
    )
    assert extra.status_code == 422

    # More than 12 negatives.
    many_neg = reverse(
        client,
        candidate_sample_ids=["C1"],
        positive_sample_ids=["P1"],
        negative_sample_ids=[f"X{i}" for i in range(13)],
    )
    assert many_neg.status_code == 422


def test_result_can_be_reread_and_listed(client):
    seed(client)
    created = reverse(
        client,
        candidate_sample_ids=["C1", "C2"],
        positive_sample_ids=["P1", "P2"],
    ).json()
    rid = created["reverse_trace_id"]

    reread = client.get(f"/reverse-traces/{rid}")
    assert reread.status_code == 200
    assert reread.json() == created

    listing = client.get("/reverse-traces").json()["reverse_traces"]
    summary = next(row for row in listing if row["reverse_trace_id"] == rid)
    assert summary["status"] == "complete"
    assert summary["solvable"] is True
    assert summary["combination_count"] == 1
    assert summary["minimum_source_count"] == 2
    assert summary["invalidated"] is False
    assert summary["candidate_sample_ids"] == ["C1", "C2"]
    assert summary["positive_sample_ids"] == ["P1", "P2"]
    assert summary["negative_sample_ids"] == []


def test_unknown_reverse_trace_is_404(client):
    assert client.get("/reverse-traces/999").status_code == 404
    assert client.get("/reverse-traces/999/invalidations").status_code == 404


def test_correction_invalidates_reverse_snapshot(client):
    seed(client)
    created = reverse(
        client,
        candidate_sample_ids=["C1"],
        positive_sample_ids=["P1"],
    ).json()
    rid = created["reverse_trace_id"]

    correction = client.post(
        "/batches/B2/corrections",
        json={
            "seen_version": 1,
            "origin": "hub",
            "destination": "lab",
            # Depart more than 24h after B1 arrived: propagation breaks.
            "departed_at": "2026-01-02T02:00:00Z",
            "arrived_at": "2026-01-02T03:00:00Z",
        },
    )
    assert correction.status_code == 200

    stale = client.get(f"/reverse-traces/{rid}")
    assert stale.status_code == 409
    assert stale.json()["code"] == "snapshot_invalidated"

    records = client.get(f"/reverse-traces/{rid}/invalidations")
    assert records.status_code == 200
    assert records.json()[0]["batch_no"] == "B2"
    assert records.json()[0]["version"] == 1

    summary = next(
        row
        for row in client.get("/reverse-traces").json()["reverse_traces"]
        if row["reverse_trace_id"] == rid
    )
    assert summary["invalidated"] is True


def test_fresh_reverse_trace_after_correction_reflects_new_values(client):
    seed(client)
    client.post(
        "/batches/B2/corrections",
        json={
            "seen_version": 1,
            "origin": "hub",
            "destination": "lab",
            "departed_at": "2026-01-02T02:00:00Z",
            "arrived_at": "2026-01-02T03:00:00Z",
        },
    )
    body = reverse(
        client,
        candidate_sample_ids=["C1"],
        positive_sample_ids=["P1"],
    ).json()
    assert body["solvable"] is False
    assert body["reason_code"] == "unexplained_positive_samples"
    assert body["snapshot"]["batch_versions"]["B2"] == 2


def test_failed_computation_publishes_nothing_and_keeps_forward_traces(
    client, monkeypatch
):
    seed(client)
    # A pre-existing forward trace and contamination flag must stay intact.
    forward = client.post("/traces", json={}).json()
    forward_id = forward["trace_id"]

    def boom(**kwargs):
        raise RuntimeError("simulated reverse failure")

    monkeypatch.setattr(reverse_module, "solve", boom)
    response = reverse(
        client,
        candidate_sample_ids=["C1"],
        positive_sample_ids=["P1"],
    )
    assert response.status_code == 500
    detail = response.json()
    assert detail["code"] == "reverse_trace_failed"
    rid = detail["details"]["reverse_trace_id"]

    # The reverse trace is failed with zero combination rows.
    read_back = client.get(f"/reverse-traces/{rid}")
    assert read_back.status_code == 500
    assert read_back.json()["code"] == "reverse_trace_failed"
    listing = client.get("/reverse-traces").json()["reverse_traces"]
    failed = next(row for row in listing if row["reverse_trace_id"] == rid)
    assert failed["status"] == "failed"
    assert failed["combination_count"] == 0

    # Forward traces and source markers are untouched.
    assert client.get(f"/traces/{forward_id}").status_code == 200
    dataset = client.get("/dataset").json()
    assert "C1" in dataset["source_sample_ids"]
    assert client.post("/traces", json={}).status_code == 201

    # A subsequent reverse request after the fault is cleared works normally.
    monkeypatch.undo()
    ok = reverse(
        client,
        candidate_sample_ids=["C1"],
        positive_sample_ids=["P1"],
    )
    assert ok.status_code == 201
    assert ok.json()["solvable"] is True


def test_reverse_failure_rolls_back_partial_combo_rows(client, monkeypatch):
    seed(client)
    real_solve = reverse_module.solve

    def corrupt(**kwargs):
        outcome = real_solve(**kwargs)
        outcome["combinations"][0]["explanations"][0]["first_arrival_at"] = "nope"
        return outcome

    monkeypatch.setattr(reverse_module, "solve", corrupt)
    response = reverse(
        client,
        candidate_sample_ids=["C1"],
        positive_sample_ids=["P1"],
    )
    assert response.status_code == 500
    rid = response.json()["details"]["reverse_trace_id"]
    with sqlite3.connect(str(client.app.state.store.path)) as conn:
        rows = conn.execute(
            "SELECT COUNT(*) FROM reverse_combos WHERE reverse_trace_id=?",
            (rid,),
        ).fetchone()[0]
    assert rows == 0


def test_crash_pending_reverse_trace_recovered(tmp_path):
    path = str(tmp_path / "crash.db")
    store = Store(path)
    store.ingest(
        samples=[
            {"sample_id": "C1", "contaminated": True, "batch_nos": []},
            {"sample_id": "P1", "contaminated": False, "batch_nos": []},
        ],
        batches=[],
        source_sample_ids=[],
    )
    rid, _snapshot = store.accept_reverse_trace(
        candidates=["C1"], positives=["P1"], negatives=[]
    )
    with sqlite3.connect(path) as conn:
        conn.execute(
            "UPDATE reverse_traces SET accepted_at='2000-01-01T00:00:00.000000+00:00'"
            " WHERE id=?",
            (rid,),
        )
    assert store.mark_interrupted() >= 1
    with TestClient(create_app(path)) as c:
        row = next(
            t
            for t in c.get("/reverse-traces").json()["reverse_traces"]
            if t["reverse_trace_id"] == rid
        )
        assert row["status"] == "failed"
        assert row["combination_count"] == 0


def test_reverse_traces_persist_across_restart(tmp_path):
    path = str(tmp_path / "persist.db")
    with TestClient(create_app(path)) as c:
        seed(c)
        created = reverse(
            c,
            candidate_sample_ids=["C1", "C2"],
            positive_sample_ids=["P1", "P2"],
        ).json()
        rid = created["reverse_trace_id"]
    with TestClient(create_app(path)) as c:
        reread = c.get(f"/reverse-traces/{rid}")
        assert reread.status_code == 200
        body = reread.json()
        assert body["minimum_source_count"] == 2
        assert body["combinations"][0]["source_sample_ids"] == ["C1", "C2"]


def test_reverse_does_not_mutate_forward_tables(client):
    seed(client)
    before = client.get("/traces").json()
    reverse(
        client,
        candidate_sample_ids=["C1"],
        positive_sample_ids=["P1"],
    )
    after = client.get("/traces").json()
    assert after == before


def test_concurrent_correction_and_reverse_never_serves_stale_advice(client):
    seed(client)
    errors: list[BaseException] = []

    def correct():
        try:
            client.post(
                "/batches/B2/corrections",
                json={
                    "seen_version": 1,
                    "origin": "hub",
                    "destination": "lab",
                    "departed_at": "2026-01-01T02:00:00.500000Z",
                    "arrived_at": "2026-01-01T03:00:00Z",
                },
            )
        except BaseException as exc:  # pragma: no cover - diagnostic
            errors.append(exc)

    thread = threading.Thread(target=correct)
    thread.start()
    response = reverse(
        client,
        candidate_sample_ids=["C1"],
        positive_sample_ids=["P1"],
    )
    thread.join()
    assert not errors

    if response.status_code == 201:
        # Either the old or the new B2 version was pinned, but never a mix.
        assert response.json()["snapshot"]["batch_versions"]["B2"] in (1, 2)
    else:
        assert response.status_code in (409, 500)
        rid = response.json()["details"]["reverse_trace_id"]
        # A failed publish leaves no readable combination rows.
        assert client.get(f"/reverse-traces/{rid}").status_code == 500
