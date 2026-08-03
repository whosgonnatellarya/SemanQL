import re

def parse_subclauses(query_string):
    

    filter_match = re.search(r'query:\s*"([^"]*)"', query_string)
    filter_string = filter_match.group(1) if filter_match else None


    first_match = re.search(r'first:\s*(\d+)', query_string)
    first_value = int(first_match.group(1)) if first_match else None


    node_match = re.search(r'node\s*{([^}]*)}', query_string)
    if node_match:
        fields = node_match.group(1).split()
    else:
        fields = []

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