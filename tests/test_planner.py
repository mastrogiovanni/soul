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
