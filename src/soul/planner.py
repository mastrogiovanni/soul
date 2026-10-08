"""Contiguous layer placement using node-computed memory offers."""

from dataclasses import dataclass

from soul.protocol import SoulError


@dataclass
class Assignment:
    node_id: str
    start: int
    end: int  # exclusive
    estimated_bytes: int


def plan(
    layers: int,
    offers: list[dict],
    max_layers_per_node: int | None = None,
    *,
    partial: bool = False,
) -> list[Assignment]:
    # Dynamic programming permits skipping nodes and accounts for both endpoint weights.
    # Stable ordering prefers larger devices and minimizes the number of network hops.
    offers = sorted(offers, key=lambda x: (x.get("order", 0), -x["budget_bytes"], x["node_id"]))
    states: dict[int, list[Assignment]] = {0: []}
    # Prefer coverage, then reuse of cached tensors, then fewer nodes. Partial
    # plans may leave gaps (e.g. an embedding stage that cannot fit yet).
    scores = {0: (0, 0, 0)}
    for offer in offers:
        missing_costs = offer.get("missing_bytes")
        updated = dict(states)
        next_scores = dict(scores)
        best = 0
        for start in range(layers):
            if start in states and scores[start] < scores[best]:
                best = start
            source = best if partial else start
            if source not in states:
                continue
            assignments = states[source]
            for end in range(start + 1, min(layers, start + (max_layers_per_node or layers)) + 1):
                cost = offer["costs"][start][end - start - 1]
                missing = missing_costs[start][end - start - 1] if missing_costs else 0
                if cost > offer["budget_bytes"] or missing > offer.get("disk_free_bytes", float("inf")):
                    continue
                candidate = assignments + [Assignment(offer["node_id"], start, end, cost)]
                score = (scores[source][0] - (end - start), scores[source][1] + missing, len(candidate))
                if end not in updated or score < next_scores[end]:
                    updated[end] = candidate
                    next_scores[end] = score
        states = updated
        scores = next_scores
    if partial:
        return states[min(states, key=scores.get)]
    if layers not in states:
        raise SoulError(
            "Insufficient node memory, disk space, or cached coverage. Preload the model, add nodes, or reduce --max-context."
        )
    return states[layers]
