from generate_queries import generate_queries
from parse_subclause import parse_subclauses
from confidence import score_consistency


def run_pipeline(merchant_question: str) -> dict:
    queries = generate_queries(merchant_question)
    parsed = [parse_subclauses(q) for q in queries]
    scores = score_consistency(parsed)
    return {
        "question": merchant_question,
        "queries": queries,
        "confidence_scores": scores
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
        print("review these before trusting the query.")
    else:
        print("\nall sub-clauses consistent across generations.")


if __name__ == "__main__":
    merchant_question = input("enter the merchant query: ")
    result = run_pipeline(merchant_question)
    print_report(result)
