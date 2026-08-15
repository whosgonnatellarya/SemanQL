from generate_queries import generate_queries
from parse_subclause import parse_subclauses
from confidence import score_consistency
from utils import pick_most_consistent
from schema_validator import validate_query

SUBCLAUSE_WEIGHT = 0.6
SCHEMA_WEIGHT = 0.4


def run_pipeline(merchant_question: str) -> dict:
    queries = generate_queries(merchant_question)
    parsed = [parse_subclauses(q) for q in queries]
    scores = score_consistency(parsed)

    best_query = pick_most_consistent(queries, parsed, scores)
    schema_result = validate_query(best_query)

    subclause_min_confidence = min(info["confidence"] for info in scores.values())
    combined_score = SUBCLAUSE_WEIGHT * subclause_min_confidence + SCHEMA_WEIGHT * schema_result["schema_score"]

    return {
        "question": merchant_question,
        "queries": queries,
        "best_query": best_query,
        "confidence_scores": scores,
        "schema_score": schema_result["schema_score"],
        "schema_violations": schema_result["violations"],
        "combined_score": combined_score,
    }


def print_report(result: dict) -> None:
    print(f"\nmerchant question: {result['question']}")
    print(f"generated {len(result['queries'])} candidate queries\n")

    print("sub-clause confidence:")
    for key, info in result["confidence_scores"].items():
        confidence = info["confidence"]
        tag = "HIGH" if confidence >= 0.8 else "LOW" if confidence < 0.6 else "MED"
        print(f"  [{tag:4}] {key:10} {confidence:.0%}  -> {info['most_common']}")

    low_confidence = [k for k, v in result["confidence_scores"].items() if v["confidence"] < 0.6]
    if low_confidence:
        print(f"\nlow confidence sub-clauses: {', '.join(low_confidence)}")
    else:
        print("\nall sub-clauses consistent across generations.")

    print(f"\nschema validation: {result['schema_score']:.0%}")
    if result["schema_violations"]:
        for v in result["schema_violations"]:
            print(f"  - {v}")
    else:
        print("no violations found.")

    combined = result["combined_score"]
    tag = "HIGH" if combined >= 0.8 else "LOW" if combined < 0.6 else "MED"
    print(f"\ncombined confidence: [{tag}] {combined:.0%}")
    if combined < 0.6:
        print("review this query before trusting it.")


if __name__ == "__main__":
    merchant_question = input("enter the merchant query: ")
    result = run_pipeline(merchant_question)
    print_report(result)
