import json
import os
import re

import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

DATASET_PATH = "dataset.jsonl"

PROMPT_TEMPLATE = """Given this merchant question: {question}
And this GraphQL query: {query}

Rate your confidence that this query correctly answers the question on a scale of 0.0 to 1.0. Return only a number."""


def self_probe(question: str, query: str) -> float:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=10,
        messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(question=question, query=query)}]
    )
    text = response.content[0].text.strip()
    match = re.search(r'\d*\.?\d+', text)
    if not match:
        raise ValueError(f"couldn't parse confidence from model response: {text!r}")
    return float(match.group())


def run_baseline(dataset_path: str = DATASET_PATH) -> None:
    with open(dataset_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    for i, entry in enumerate(entries, start=1):
        print(f"[{i}/{len(entries)}] self-probing: {entry['question']}")
        entry["self_probing_confidence"] = self_probe(entry["question"], entry["generated_query"])

    with open(dataset_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    print(f"\nupdated {len(entries)} entries in {dataset_path}")


if __name__ == "__main__":
    run_baseline()
