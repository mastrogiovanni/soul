"""Contiguous layer placement using node-computed memory offers."""

from dataclasses import dataclass

from soul.protocol import SoulError


@dataclass
class Assignment:
    node_id: str
    start: int
    end: int  # exclusive
    estimated_bytes: int


def plan(layers: int, offers: list[dict], max_layers_per_node: int | None = None) -> list[Assignment]:
    # Dynamic programming permits skipping nodes and accounts for both endpoint weights.
    # Stable ordering prefers larger devices and minimizes the number of network hops.
    offers = sorted(offers, key=lambda x: (-x["budget_bytes"], x["node_id"]))
    states: dict[int, list[Assignment]] = {0: []}
    for offer in offers:
        updated = dict(states)
        for start, assignments in states.items():
            for end in range(start + 1, min(layers, start + (max_layers_per_node or layers)) + 1):
                cost = offer["costs"][start][end - start - 1]
                if cost > offer["budget_bytes"]:
                    continue
                candidate = assignments + [Assignment(offer["node_id"], start, end, cost)]
                if end not in updated or len(candidate) < len(updated[end]):
                    updated[end] = candidate
        states = updated
    if layers not in states:
        raise SoulError(
            "Insufficient node memory for weights, KV cache and workspace. Add nodes or reduce --max-context."
        )
    return states[layers]
