def pick_most_consistent(queries: list[str], parsed: list[dict], scores: dict) -> str:
    """pick the generated query whose sub-clauses match the most_common value most often"""
    best_idx = 0
    best_match = -1
    for i, p in enumerate(parsed):
        match_count = 0
        for key, info in scores.items():
            value = p[key]
            if isinstance(value, list):
                value = tuple(value)
            if value == info["most_common"]:
                match_count += 1
        if match_count > best_match:
            best_match = match_count
            best_idx = i
    return queries[best_idx]
