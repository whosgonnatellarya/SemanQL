import json
import os

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

# batch 2: straightforward questions covering date ranges, sortKey-based
# sorting, multiple combined filters, and pagination
MERCHANT_QUESTIONS_V2 = [
    "get orders placed between January 1 and March 31 this year",
    "find orders created in the last 7 days",
    "get orders sorted by total price, highest first",
    "find customers sorted by amount spent, highest first",
    "get the next 50 orders after a given cursor",
    "find customers from the United States who have spent over $200",
    "get orders sorted by order number",
    "find customers tagged as newsletter who spent over $50",
    "get orders between $50 and $200 sorted by creation date",
    "find customers created between June 1 and June 30",
    "get the last 25 orders sorted by update date",
    "find customers with more than 2 orders and tagged as wholesale",
    "get fulfilled orders sorted by processed date",
    "find customers from Canada or the United States",
    "get orders sorted by relevance for the search term 'blue shirt'",
    "find customers who spent over $100 and have a verified email",
    "get paid orders from the last 90 days",
    "find the next page of customers after the first 50",
    "get orders tagged 'gift' sorted by creation date, oldest first",
    "find customers not from the United States who spent over $1000",
]

# batch 2: designed to trigger semantic errors -- wrong filter keys,
# deprecated field names, plausible but nonexistent fields
SEMANTIC_ERROR_QUESTIONS_V2 = [
    "find customers by loyalty tier",                          # no loyalty field in this schema
    "get orders using the old total price field",               # deprecated total_price vs totalPriceSet
    "find customers whose account is enabled",                  # tag:enabled trap vs customer_account_status
    "get orders sorted by the deprecated financial status field",  # financialStatus vs displayFinancialStatus
    "find customers with a risk score above 50",                 # no risk field exists
    "get orders that still have a balance due",                  # no balance-due field, only financial status
    "find customers who opted into marketing",                   # no marketing consent field exposed
    "get orders that used a discount code",                      # no discount field exposed
    "find customers by loyalty points",                          # loyalty points don't exist in this schema
    "get orders shipped via express shipping",                   # no shipping method field exposed
    "find customers flagged as fraudulent",                      # no fraud field exists
    "get orders with a gift card applied",                       # no gift card field exposed
    "find customers subscribed to sms",                          # sms consent not in this schema
    "get orders sorted by the old fulfillment status field",     # fulfillmentStatus vs displayFulfillmentStatus
    "find customers by lifetime value tier",                     # undefined field/threshold
    "get orders using the legacy total field before totalPriceSet",  # deprecated field name trap
    "find customers who are blocked",                            # "blocked" isn't a real account status value
    "get orders that are on hold",                                # no hold status in this schema
    "find customers with an open support ticket",                 # no ticket data in this schema
    "get orders using the old order status enum",                 # deprecated status enum trap
]

# batch 2: edge cases -- valid but unusual syntax, nested filters, and
# multiple AND/OR conditions
EDGE_CASE_QUESTIONS_V2 = [
    "find customers from Canada or the United States who spent over $300",
    "get orders that are either cancelled or refunded, placed this year",
    "find customers tagged wholesale or vip who have more than 10 orders",
    "get orders over $500 AND paid AND fulfilled",
    "find customers with spent over $1000 or more than 20 orders, and from the US",
    "get orders that are not tagged as test",
    "find customers whose email ends in .edu",
    "get orders sorted by total price descending, limited to the first 5",
    "find customers created after 2024 but before 2025, tagged newsletter",
    "get the second page of orders using a cursor, sorted by processed date",
]

ALL_QUESTIONS = (
    MERCHANT_QUESTIONS
    + SEMANTIC_ERROR_QUESTIONS
    + MERCHANT_QUESTIONS_V2
    + SEMANTIC_ERROR_QUESTIONS_V2
    + EDGE_CASE_QUESTIONS_V2
)


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


def _load_existing(output_path: str) -> list[dict]:
    if not os.path.exists(output_path):
        return []
    with open(output_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_dataset(questions: list[str] = ALL_QUESTIONS, output_path: str = OUTPUT_PATH) -> None:
    """generates entries for any question not already in output_path and appends
    them, leaving existing entries (labels, self-probing scores) untouched"""
    entries = _load_existing(output_path)
    seen = {e["question"] for e in entries}
    new_questions = [q for q in questions if q not in seen]

    if not new_questions:
        print("no new questions to generate.")
        return

    for i, question in enumerate(new_questions, start=1):
        print(f"[{i}/{len(new_questions)}] generating: {question}")
        queries = generate_queries(question)
        parsed = [parse_subclauses(q) for q in queries]
        scores = score_consistency(parsed)
        best_query = pick_most_consistent(queries, parsed, scores)

        entries.append({
            "question": question,
            "generated_query": best_query,
            "sub_clause_confidence": {k: v["confidence"] for k, v in scores.items()},
            "label": None
        })

        # save after every entry so an interrupted run loses no progress --
        # a rerun will just pick up with whatever's left in new_questions
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

    print(f"\nappended {len(new_questions)} new entries to {output_path} ({len(entries)} total)")


if __name__ == "__main__":
    build_dataset()
