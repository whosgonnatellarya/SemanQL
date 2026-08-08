from collections import Counter

from generate_queries import generate_queries
from parse_subclause import parse_subclauses


def score_consistency(results):
    """results = list of 5 dicts from parse_subclauses"""
    scores = {}

    for key in ["filter", "first", "fields", "pageInfo"]:
        values = []
        for r in results:
            v = r[key]
            if isinstance(v, list):
                v = tuple(v)  # lists aren't hashable, tuples are
            values.append(v)

        counts = Counter(values)
        most_common_value, most_common_count = counts.most_common(1)[0]
        scores[key] = {
            "confidence": most_common_count / len(results),
            "most_common": most_common_value
        }

    return scores


if __name__ == "__main__":
    merchant_question = input("enter the merchant query: ")
    queries = generate_queries(merchant_question)
    results = [parse_subclauses(q) for q in queries]
    scores = score_consistency(results)
    print(scores)