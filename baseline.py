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

Rate your confidence that this query correctly answers the question on a scale of 0.0 to 1.0. Respond with ONLY the number, e.g. "0.7" -- no words, no explanation."""

MAX_ATTEMPTS = 3


def self_probe(question: str, query: str) -> float:
    prompt = PROMPT_TEMPLATE.format(question=question, query=query)
    for attempt in range(MAX_ATTEMPTS):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.content[0].text.strip()
        match = re.search(r'\d*\.?\d+', text)
        if match:
            return float(match.group())
        if attempt < MAX_ATTEMPTS - 1:
            print(f"  (retrying, unparseable response: {text!r})")
    raise ValueError(f"couldn't parse confidence from model response after {MAX_ATTEMPTS} attempts: {text!r}")


def run_baseline(dataset_path: str = DATASET_PATH) -> None:
    with open(dataset_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    todo = [e for e in entries if "self_probing_confidence" not in e]
    if not todo:
        print("all entries already have self-probing scores.")
        return

    for i, entry in enumerate(todo, start=1):
        print(f"[{i}/{len(todo)}] self-probing: {entry['question']}")
        entry["self_probing_confidence"] = self_probe(entry["question"], entry["generated_query"])

        # save after every entry so an interrupted run loses no progress
        with open(dataset_path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

    print(f"\nupdated {len(todo)} entries in {dataset_path}")


if __name__ == "__main__":
    run_baseline()
