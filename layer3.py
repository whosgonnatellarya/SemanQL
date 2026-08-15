import json
import os
import re

import anthropic
from dotenv import load_dotenv

from generate_queries import generate_queries
from parse_subclause import parse_subclauses
from confidence import score_consistency
from utils import pick_most_consistent

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

PARAPHRASE_PROMPT_TEMPLATE = """Rephrase this shopify merchant question in a different way but with exactly the same intent. Return only the rephrased question, nothing else.

Question: {question}"""

EQUIVALENCE_PROMPT_TEMPLATE = """Given this merchant question: {question}

Query 1: {query1}

Query 2: {query2}

Are these two queries semantically equivalent, meaning would they return the same data from a shopify store? Answer with a score from 0.0 (completely different) to 1.0 (identical meaning) and a one-sentence explanation. Return ONLY a JSON object, no markdown formatting, no explanation outside the JSON: {{"score": float, "explanation": str}}"""

MAX_ATTEMPTS = 3


def paraphrase_question(question: str) -> str:
    """ask Claude to rephrase the merchant question differently but with the same intent"""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        messages=[{"role": "user", "content": PARAPHRASE_PROMPT_TEMPLATE.format(question=question)}]
    )
    return response.content[0].text.strip()


def get_most_consistent_query(question: str) -> tuple[str, dict]:
    """run the full pipeline on a question, return the most consistent query and its scores"""
    queries = generate_queries(question)
    parsed = [parse_subclauses(q) for q in queries]
    scores = score_consistency(parsed)
    best_query = pick_most_consistent(queries, parsed, scores)
    return best_query, scores


def check_semantic_equivalence(query1: str, query2: str, question: str) -> dict:
    """use Claude to judge whether two queries are semantically equivalent.

    returns {"score": float in [0, 1], "explanation": str}
    """
    prompt = EQUIVALENCE_PROMPT_TEMPLATE.format(question=question, query1=query1, query2=query2)
    text = None
    for attempt in range(MAX_ATTEMPTS):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.content[0].text.strip()
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                if "score" in result and "explanation" in result:
                    return {"score": float(result["score"]), "explanation": str(result["explanation"])}
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        if attempt < MAX_ATTEMPTS - 1:
            print(f"  (retrying, unparseable equivalence response: {text!r})")
    raise ValueError(f"couldn't parse equivalence json from model response after {MAX_ATTEMPTS} attempts: {text!r}")


def run_layer3(question: str) -> dict:
    """full layer 3 analysis: paraphrase the question, run the pipeline on both the
    original and the paraphrase independently, and check whether the two resulting
    most-consistent queries are semantically equivalent. a low equivalence score means
    the model answered two equivalent questions two different ways -- a signal that
    layers 1 (consistent-but-wrong) and 2 (schema-valid-but-wrong) can both miss."""
    original_query, _original_scores = get_most_consistent_query(question)
    paraphrase = paraphrase_question(question)
    paraphrase_query, _paraphrase_scores = get_most_consistent_query(paraphrase)
    equivalence = check_semantic_equivalence(original_query, paraphrase_query, question)

    return {
        "question": question,
        "paraphrase": paraphrase,
        "original_query": original_query,
        "paraphrase_query": paraphrase_query,
        "equivalence_score": equivalence["score"],
        "explanation": equivalence["explanation"],
    }


if __name__ == "__main__":
    merchant_question = input("enter the merchant query: ")
    result = run_layer3(merchant_question)
    print(f"\nparaphrase: {result['paraphrase']}")
    print(f"\noriginal query:\n{result['original_query']}")
    print(f"\nparaphrase query:\n{result['paraphrase_query']}")
    print(f"\nequivalence score: {result['equivalence_score']:.2f}")
    print(f"explanation: {result['explanation']}")
