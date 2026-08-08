import json

from generate_queries import generate_queries
from parse_subclause import parse_subclauses
from confidence import score_consistency

OUTPUT_PATH = "dataset.jsonl"

# straightforward / varying complexity
MERCHANT_QUESTIONS = [
    "find customers who spent over $500",
    "get the 20 most recent orders",
    "find customers from Canada",
    "get orders that are still unfulfilled",
    "find customers who have placed more than 5 orders",
    "get cancelled orders from this year",
    "find customers whose email contains gmail.com",
    "get orders with a pending payment status",
    "find customers created in the last 30 days",
    "get the first 100 orders sorted by creation date",
    "find customers tagged as wholesale",
    "get orders tagged as rush",
    "find enabled customers who have placed 3 or more orders",
    "get orders over $1000 that are paid",
    "find customers who spent between $100 and $500",
]

# designed to be ambiguous or reference non-existent / mis-mapped fields,
# so the model is likely to generate a semantically wrong query
SEMANTIC_ERROR_QUESTIONS = [
    "find customers tagged as enabled",          # should use customer_account_status, not tags
    "get orders from last month",                # ambiguous relative date filtering
    "find vip customers",                        # vip isn't a real field
    "customers who haven't ordered recently",    # no clear field for recency of absence
    "find customers who are likely to churn",    # no churn field exists
    "get orders that were refunded",             # no explicit refund field, only financial status
    "find high value customers",                 # "high value" undefined threshold/field
    "customers who abandoned their cart",        # no cart data in this schema at all
    "get orders shipped internationally",        # no shipping-country field exposed
    "find customers who are repeat buyers",      # "repeat" undefined, could be numberOfOrders > 1 or > 0
]

ALL_QUESTIONS = MERCHANT_QUESTIONS + SEMANTIC_ERROR_QUESTIONS


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


def build_dataset(questions: list[str] = ALL_QUESTIONS, output_path: str = OUTPUT_PATH) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for i, question in enumerate(questions, start=1):
            print(f"[{i}/{len(questions)}] generating: {question}")
            queries = generate_queries(question)
            parsed = [parse_subclauses(q) for q in queries]
            scores = score_consistency(parsed)
            best_query = pick_most_consistent(queries, parsed, scores)

            entry = {
                "question": question,
                "generated_query": best_query,
                "sub_clause_confidence": {k: v["confidence"] for k, v in scores.items()},
                "label": None
            }
            f.write(json.dumps(entry) + "\n")

    print(f"\nwrote {len(questions)} entries to {output_path}")


if __name__ == "__main__":
    build_dataset()
