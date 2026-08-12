import json

import polars as pl
from sklearn.metrics import roc_auc_score

from schema_validator import validate_query

DATASET_PATH = "dataset.jsonl"
RESULTS_PATH = "evaluation_results.json"

SUBCLAUSE_WEIGHT = 0.6
SCHEMA_WEIGHT = 0.4

# every signal below is a "wrongness" score: higher = more likely the query is
# semantically wrong. label is flipped to match (1 = wrong, 0 = correct), so
# a higher auc always means "better at catching wrong queries" for every method.


def load_labeled_entries(path: str = DATASET_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    return [e for e in entries if e.get("label") is not None]


def build_signal_frame(entries: list[dict]) -> pl.DataFrame:
    rows = []
    for e in entries:
        min_confidence = min(e["sub_clause_confidence"].values())
        schema_score = validate_query(e["generated_query"])["schema_score"]
        combined_score = SUBCLAUSE_WEIGHT * min_confidence + SCHEMA_WEIGHT * schema_score

        rows.append({
            "question": e["question"],
            "wrong": 1 - e["label"],
            "subclause_wrongness": 1 - min_confidence,
            "self_probing_wrongness": 1 - e["self_probing_confidence"],
            "schema_wrongness": 1 - schema_score,
            "combined_wrongness": 1 - combined_score,
        })
    return pl.DataFrame(rows)


def run_evaluation(path: str = DATASET_PATH, results_path: str = RESULTS_PATH) -> dict:
    entries = load_labeled_entries(path)
    if not entries:
        print("no labeled entries found. run label_dataset.py first.")
        return {}

    df = build_signal_frame(entries)
    wrong = df["wrong"].to_list()

    methods = {
        "sub_clause_frequency": "subclause_wrongness",
        "self_probing_baseline": "self_probing_wrongness",
        "schema_validation": "schema_wrongness",
        "combined": "combined_wrongness",
    }

    results = {"n_labeled": len(entries), "methods": {}}
    print(f"\nevaluated on {len(entries)} labeled entries\n")
    print(f"{'method':22} {'auc':>6}")
    for name, column in methods.items():
        auc = roc_auc_score(wrong, df[column].to_list())
        results["methods"][name] = auc
        print(f"{name:22} {auc:.3f}")

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved results to {results_path}")

    return results


if __name__ == "__main__":
    run_evaluation()
