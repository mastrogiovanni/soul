import pytest

from soul.controller.engine import TextBuffer
from soul.planner import plan
from soul.protocol import SoulError


def offer(id, budget, layers=10, edge=0):
    return {
        "node_id": id,
        "budget_bytes": budget,
        "costs": [
            [
                (end - start) * 10 + (edge if start == 0 else 0) + (edge if end == layers else 0)
                for end in range(start + 1, layers + 1)
            ]
            for start in range(layers)
        ],
    }


def test_placement_covers_every_layer_exactly_once():
    assignments = plan(10, [offer("a", 40), offer("b", 30), offer("c", 30)])
    assert [(a.start, a.end) for a in assignments] == [(0, 4), (4, 7), (7, 10)]


def test_edges_reserve_memory_and_fail_when_missing():
    with pytest.raises(SoulError):
        plan(10, [offer("a", 60, edge=30), offer("b", 60, edge=30)])
    result = plan(10, [offer("a", 80, edge=30), offer("b", 80, edge=30)])
    assert result[-1].end == 10


def test_prefer_fewer_hops_and_honor_shard_limit():
    assert len(plan(10, [offer("a", 200), offer("b", 200)])) == 1
    assert len(plan(10, [offer("a", 200), offer("b", 200)], 5)) == 2


def test_partial_plan_caches_what_fits_and_strict_plan_still_rejects_gaps():
    offers = [offer("a", 40), offer("b", 30)]
    result = plan(10, offers, partial=True)
    assert [(a.start, a.end) for a in result] == [(0, 4), (4, 7)]
    assert len({a.node_id for a in result}) == len(result)
    with pytest.raises(SoulError):
        plan(10, offers)
    assert plan(10, [], partial=True) == []


def test_partial_plan_handles_missing_endpoint_capacity_and_disk_limits():
    # Neither edge stage fits, but middle layers can still be cached usefully.
    offers = [offer("a", 30, layers=6, edge=30), offer("b", 30, layers=6, edge=30)]
    for item in offers:
        item["missing_bytes"] = [[10 * (i + 1) for i in range(len(row))] for row in item["costs"]]
        item["disk_free_bytes"] = 20
    result = plan(6, offers, partial=True)
    assert [(a.start, a.end) for a in result] == [(1, 3), (3, 5)]
    assert plan(6, [{**offers[0], "disk_free_bytes": 0}], partial=True) == []
    assert sum(a.end - a.start for a in plan(6, offers, 1, partial=True)) == 2


def test_partial_plan_preserves_cached_suffix_and_fills_gap_with_new_node():
    cached = offer("cached", 30, layers=6)
    cached["missing_bytes"] = [[10 * (i + 1) for i in range(len(row))] for row in cached["costs"]]
    for start in range(3, 6):
        cached["missing_bytes"][start] = [0] * (6 - start)
    cached["disk_free_bytes"] = 0
    assert [(a.start, a.end) for a in plan(6, [cached], partial=True)] == [(3, 6)]
    new = offer("new", 30, layers=6)
    new["missing_bytes"] = [[10 * (i + 1) for i in range(len(row))] for row in new["costs"]]
    result = plan(6, [new, {**cached, "order": 3}], partial=True)
    assert [(a.node_id, a.start, a.end) for a in result] == [("new", 0, 3), ("cached", 3, 6)]


def test_stop_split_across_tokens_is_never_emitted():
    buffer = TextBuffer(["<END>"])
    assert buffer.update("hello <E") == "hello "
    assert buffer.update("hello <END> trailing") == ""
    assert buffer.matched == "<END>"


def test_partial_unicode_and_final_stop_prefix():
    buffer = TextBuffer(["END"])
    assert buffer.update("hi \ufffd") == "hi "
    assert buffer.update("hi 😀 E") == "😀 "
    assert buffer.update("hi 😀 E", final=True) == "E"
