import re

from parse_subclause import parse_subclauses

# hardcoded shopify admin api field lists (customers + orders), good enough
# for the query shapes this project generates. not the full schema.
VALID_CUSTOMER_FIELDS = {
    "id", "email", "firstName", "lastName", "amountSpent", "numberOfOrders",
    "createdAt", "updatedAt", "tags", "state", "displayName", "phone", "note",
    "verifiedEmail", "defaultAddress", "customer_account_status", "locale",
}

VALID_ORDER_FIELDS = {
    "id", "name", "createdAt", "updatedAt", "processedAt", "totalPriceSet",
    "currentTotalPriceSet", "displayFinancialStatus", "displayFulfillmentStatus",
    "cancelledAt", "tags", "customer", "email", "note", "closed", "test",
}

# structural / pagination fields that are valid wherever they appear
VALID_STRUCTURAL_FIELDS = {"edges", "node", "pageInfo", "hasNextPage", "hasPreviousPage", "endCursor", "startCursor", "cursor"}

VALID_FIELDS = VALID_CUSTOMER_FIELDS | VALID_ORDER_FIELDS | VALID_STRUCTURAL_FIELDS

VALID_FILTER_KEYS = {"state", "amount_spent", "email", "tag", "country", "created_at"}

# ordered longest-first so ">=" isn't mistaken for ">"
VALID_OPERATORS = [":>=", ":<=", ":>", ":<", ":"]

MIN_FIRST = 1
MAX_FIRST = 250


def _split_filter_clauses(filter_string: str) -> list[str]:
    return [c.strip() for c in re.split(r"\s+(?:AND|OR)\s+", filter_string) if c.strip()]


def _check_filter_clause(clause: str) -> tuple[bool, str | None]:
    for op in VALID_OPERATORS:
        if op in clause:
            key = clause.split(op, 1)[0].strip()
            if key not in VALID_FILTER_KEYS:
                return False, f"unknown filter key: {key!r} in clause {clause!r}"
            return True, None
    return False, f"no valid operator found in filter clause {clause!r}"


def validate_query(query_string: str) -> dict:
    """checks a generated query against the known shopify schema.

    returns {"schema_score": float in [0, 1], "violations": [str, ...]}
    score is the fraction of individual checks that passed.
    """
    parsed = parse_subclauses(query_string)
    violations = []
    checks_passed = 0
    checks_total = 0

    # field checks
    for field in parsed["fields"]:
        checks_total += 1
        if field in VALID_FIELDS:
            checks_passed += 1
        else:
            violations.append(f"unknown field: {field!r}")

    # filter checks (key + operator per clause)
    if parsed["filter"]:
        for clause in _split_filter_clauses(parsed["filter"]):
            checks_total += 1
            ok, reason = _check_filter_clause(clause)
            if ok:
                checks_passed += 1
            else:
                violations.append(reason)

    # first: range check
    if parsed["first"] is not None:
        checks_total += 1
        if MIN_FIRST <= parsed["first"] <= MAX_FIRST:
            checks_passed += 1
        else:
            violations.append(f"first: {parsed['first']} out of range [{MIN_FIRST}, {MAX_FIRST}]")

    # no checkable sub-clauses at all is itself suspicious for a query
    if checks_total == 0:
        return {"schema_score": 0.0, "violations": ["no recognizable fields, filter, or first: found"]}

    return {
        "schema_score": checks_passed / checks_total,
        "violations": violations,
    }


if __name__ == "__main__":
    test_query = '''
    {
      customers(first: 50, query: "state:enabled AND amount_spent:>500") {
        edges {
          node {
            id
            email
            vipStatus
          }
        }
      }
    }
    '''
    print(validate_query(test_query))
