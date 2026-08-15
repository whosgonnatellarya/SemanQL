import re

from parse_subclause import parse_subclauses

# hardcoded shopify admin api field lists (customers + orders), good enough
# for the query shapes this project generates. not the full schema.
VALID_CUSTOMER_FIELDS = {
    "id", "email", "firstName", "lastName", "amountSpent", "numberOfOrders",
    "createdAt", "updatedAt", "tags", "state", "displayName", "phone", "note",
    "verifiedEmail", "defaultAddress", "locale",
}

VALID_ORDER_FIELDS = {
    "id", "name", "createdAt", "updatedAt", "processedAt", "totalPriceSet",
    "currentTotalPriceSet", "displayFinancialStatus", "displayFulfillmentStatus",
    "cancelledAt", "tags", "customer", "email", "note", "closed", "test",
}

# structural / pagination fields that are valid wherever they appear
VALID_STRUCTURAL_FIELDS = {"edges", "node", "pageInfo", "hasNextPage", "hasPreviousPage", "endCursor", "startCursor", "cursor"}

VALID_FIELDS = VALID_CUSTOMER_FIELDS | VALID_ORDER_FIELDS | VALID_STRUCTURAL_FIELDS

# filter keys are context-dependent -- a key valid on customers() isn't
# necessarily valid on orders() and vice versa, so these are kept separate
# and the right one is picked per-query (see _filter_keys_for_query)
CUSTOMER_FILTER_KEYS = {
    "accepts_marketing", "country", "customer_date", "email", "first_name", "id",
    "last_abandoned_order_date", "last_name", "order_date", "orders_count", "phone",
    "state", "tag", "tag_not", "total_spent", "updated_at",
}

ORDER_FILTER_KEYS = {
    "financial_status", "fulfillment_status", "status", "created_at", "updated_at",
    "total_price", "email", "customer_id", "tag",
}

# value-level enums / rules for specific filter keys
STATE_VALUES = {"ENABLED", "INVITED", "DISABLED", "DECLINED"}
FINANCIAL_STATUS_VALUES = {"pending", "authorized", "partially_paid", "paid", "partially_refunded", "refunded", "voided"}
FULFILLMENT_STATUS_VALUES = {"shipped", "partial", "unshipped", "unfulfilled"}
ORDER_STATUS_VALUES = {"open", "closed", "cancelled", "any"}

# filter keys whose values are dates and must be quoted with single quotes
DATE_FILTER_KEYS = {"created_at", "customer_date", "updated_at", "order_date", "last_abandoned_order_date"}

# ordered longest-first so ">=" isn't mistaken for ">"
VALID_OPERATORS = [":>=", ":<=", ":>", ":<", ":"]

MIN_FIRST = 1
MAX_FIRST = 250


def _filter_keys_for_query(query_string: str) -> set[str]:
    """picks the right filter-key set based on whether this is a customers() or
    orders() query -- a key valid on one isn't necessarily valid on the other.
    if neither root is recognized, falls back to the union (better to under- than
    over-flag on a query shape this project doesn't expect)."""
    if re.search(r'\bcustomers\s*\(', query_string):
        return CUSTOMER_FILTER_KEYS
    if re.search(r'\borders\s*\(', query_string):
        return ORDER_FILTER_KEYS
    return CUSTOMER_FILTER_KEYS | ORDER_FILTER_KEYS


def _strip_negation(clause: str) -> str:
    """strips a leading 'NOT ' or '-' negation prefix so it doesn't get treated
    as part of the field name (e.g. 'NOT tag:test' -> 'tag:test')"""
    if clause.startswith("NOT "):
        return clause[len("NOT "):].strip()
    if clause.startswith("-") and len(clause) > 1 and (clause[1].isalpha() or clause[1] == "_"):
        return clause[1:].strip()
    return clause


def _split_filter_clauses(filter_string: str) -> list[str]:
    # split on explicit AND/OR, and also before a bare NOT that's acting as an
    # implicit conjunction between two clauses (e.g. "total_spent:>1000 NOT
    # country:US" is really "total_spent:>1000 AND (NOT country:US)")
    parts = re.split(r"\s+(?:AND|OR)\s+|\s+(?=NOT\s)", filter_string)
    return [c.strip() for c in parts if c.strip()]


def _check_value(key: str, value: str, clause: str) -> str | None:
    """returns a violation message if the value is invalid for this key, else None"""
    if key == "state" and value not in STATE_VALUES:
        return f"state value must be uppercase, one of {sorted(STATE_VALUES)}: got {value!r} in clause {clause!r}"

    if key == "financial_status" and value not in FINANCIAL_STATUS_VALUES:
        return f"financial_status value must be one of {sorted(FINANCIAL_STATUS_VALUES)}: got {value!r} in clause {clause!r}"

    if key == "fulfillment_status" and value not in FULFILLMENT_STATUS_VALUES:
        return f"fulfillment_status value must be one of {sorted(FULFILLMENT_STATUS_VALUES)}: got {value!r} in clause {clause!r}"

    if key == "status" and value not in ORDER_STATUS_VALUES:
        return f"status value must be one of {sorted(ORDER_STATUS_VALUES)}: got {value!r} in clause {clause!r}"

    if key in DATE_FILTER_KEYS and not (len(value) >= 2 and value.startswith("'") and value.endswith("'")):
        return f"date value must be quoted with single quotes: got {value!r} in clause {clause!r}"

    return None


def _check_filter_clause(clause: str, valid_filter_keys: set[str]) -> tuple[bool, str | None]:
    clause = _strip_negation(clause.strip())

    for op in VALID_OPERATORS:
        if op in clause:
            key, _, value = clause.partition(op)
            key = key.strip()
            value = value.strip()

            if key not in valid_filter_keys:
                return False, f"unknown filter key: {key!r} in clause {clause!r}"

            violation = _check_value(key, value, clause)
            if violation is not None:
                return False, violation

            return True, None

    return False, f"no valid operator found in filter clause {clause!r}"


def validate_query(query_string: str) -> dict:
    """checks a generated query against the known shopify schema.

    returns {"schema_score": float in [0, 1], "violations": [str, ...]}
    score is the fraction of individual checks that passed.
    """
    parsed = parse_subclauses(query_string)
    valid_filter_keys = _filter_keys_for_query(query_string)
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

    # filter checks (key + operator + value per clause)
    if parsed["filter"]:
        for clause in _split_filter_clauses(parsed["filter"]):
            checks_total += 1
            ok, reason = _check_filter_clause(clause, valid_filter_keys)
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
      customers(first: 50, query: "state:enabled AND total_spent:>500") {
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
