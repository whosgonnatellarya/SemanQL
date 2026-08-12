import re

def _extract_node_body(query_string):
    """find the node { ... } block and return its contents, respecting nested braces
    (a naive [^}]* regex stops at the first '}', mangling nested selections like
    amountSpent { amount currencyCode })"""
    open_match = re.search(r'node\s*\{', query_string)
    if not open_match:
        return None

    depth = 1
    i = open_match.end()
    while i < len(query_string) and depth > 0:
        if query_string[i] == '{':
            depth += 1
        elif query_string[i] == '}':
            depth -= 1
        i += 1
    return query_string[open_match.end():i - 1]


def _top_level_fields(body):
    """field names at the top level of a selection set, skipping nested sub-selections"""
    fields = []
    depth = 0
    for tok in re.findall(r'\{|\}|\w+', body):
        if tok == '{':
            depth += 1
        elif tok == '}':
            depth -= 1
        elif depth == 0:
            fields.append(tok)
    return fields


def parse_subclauses(query_string):


    filter_match = re.search(r'query:\s*"([^"]*)"', query_string)
    filter_string = filter_match.group(1) if filter_match else None


    first_match = re.search(r'first:\s*(\d+)', query_string)
    first_value = int(first_match.group(1)) if first_match else None


    node_body = _extract_node_body(query_string)
    fields = _top_level_fields(node_body) if node_body is not None else []

    has_page_info = 'pageInfo' in query_string

    return {
        "filter": filter_string,
        "first": first_value,
        "fields": sorted(fields),  # sorted so order differences don't matter
        "pageInfo": has_page_info
    }


if __name__ == "__main__":
    test_query = '''
    {
      customers(first: 50, query: "state:enabled AND amount_spent:>500") {
        edges {
          node {
            id
            email
            firstName
            lastName
          }
        }
      }
    }
    '''
    result = parse_subclauses(test_query)
    print(result)