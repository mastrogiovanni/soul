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
    offers = sorted(offers, key=lambda x: (x.get("order", 0), -x["budget_bytes"], x["node_id"]))
    states: dict[int, list[Assignment]] = {0: []}
    scores = {0: (0, 0)}
    for offer in offers:
        missing_costs = offer.get("missing_bytes")
        updated = dict(states)
        next_scores = dict(scores)
        for start, assignments in states.items():
            for end in range(start + 1, min(layers, start + (max_layers_per_node or layers)) + 1):
                cost = offer["costs"][start][end - start - 1]
                missing = missing_costs[start][end - start - 1] if missing_costs else 0
                if cost > offer["budget_bytes"] or missing > offer.get("disk_free_bytes", float("inf")):
                    continue
                candidate = assignments + [Assignment(offer["node_id"], start, end, cost)]
                score = (scores[start][0] + missing, len(candidate))
                if end not in updated or score < next_scores[end]:
                    updated[end] = candidate
                    next_scores[end] = score
        states = updated
        scores = next_scores
    if layers not in states:
        raise SoulError(
            "Insufficient node memory, disk space, or cached coverage. Preload the model, add nodes, or reduce --max-context."
        )
    return states[layers]
