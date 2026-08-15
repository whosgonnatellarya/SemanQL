import argparse
import json

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from schema_validator import validate_query
from layer3 import paraphrase_question, get_most_consistent_query, check_semantic_equivalence

DATASET_PATH = "dataset.jsonl"
RESULTS_PATH = "evaluation_results.json"

SUBCLAUSE_WEIGHT = 0.6
SCHEMA_WEIGHT = 0.4

ALL_LAYERS_SUBCLAUSE_WEIGHT = 0.4
ALL_LAYERS_SCHEMA_WEIGHT = 0.3
ALL_LAYERS_LAYER3_WEIGHT = 0.3

N_BOOTSTRAP = 1000
CI = 0.95
BOOTSTRAP_SEED = 42

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
        combined_l1_l2 = SUBCLAUSE_WEIGHT * min_confidence + SCHEMA_WEIGHT * schema_score

        row = {
            "question": e["question"],
            "wrong": 1 - e["label"],
            "subclause_wrongness": 1 - min_confidence,
            "self_probing_wrongness": 1 - e["self_probing_confidence"],
            "schema_wrongness": 1 - schema_score,
            "combined_l1_l2_wrongness": 1 - combined_l1_l2,
        }

        if "layer3_equivalence" in e:
            equivalence = e["layer3_equivalence"]
            # equivalence_score (not 1 - equivalence_score) here so all three terms
            # are confidence-style, matching how combined_l1_l2 is built -- the whole
            # sum is inverted once below, same as combined_l1_l2 is
            combined_all_layers = (
                ALL_LAYERS_SUBCLAUSE_WEIGHT * min_confidence
                + ALL_LAYERS_SCHEMA_WEIGHT * schema_score
                + ALL_LAYERS_LAYER3_WEIGHT * equivalence
            )
            row["semantic_equivalence_wrongness"] = 1 - equivalence
            row["combined_all_layers_wrongness"] = 1 - combined_all_layers

        rows.append(row)
    return pl.DataFrame(rows)


def populate_layer3_scores(path: str = DATASET_PATH) -> None:
    """runs layer 3 (paraphrase + cross-check) on any dataset entry missing an
    equivalence score. reuses the entry's already-generated `generated_query` for
    the original side instead of regenerating it -- dataset entries already store
    the most-consistent query from the same generate+score pipeline
    layer3.get_most_consistent_query would otherwise redo, so this roughly halves
    the API calls needed per entry. saves progress after every entry so an
    interrupted run doesn't lose work."""
    with open(path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    todo = [e for e in entries if "layer3_equivalence" not in e]
    if not todo:
        print("all entries already have layer 3 scores.")
        return

    for i, entry in enumerate(todo, start=1):
        print(f"[{i}/{len(todo)}] layer 3: {entry['question']}")
        question = entry["question"]
        original_query = entry["generated_query"]

        paraphrase = paraphrase_question(question)
        paraphrase_query, _ = get_most_consistent_query(paraphrase)
        equivalence = check_semantic_equivalence(original_query, paraphrase_query, question)

        entry["layer3_paraphrase"] = paraphrase
        entry["layer3_paraphrase_query"] = paraphrase_query
        entry["layer3_equivalence"] = equivalence["score"]
        entry["layer3_explanation"] = equivalence["explanation"]

        with open(path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

    print(f"\nupdated {len(todo)} entries with layer 3 scores in {path}")


def bootstrap_auc_ci(
    y_true: list[int], y_score: list[float],
    n_bootstrap: int = N_BOOTSTRAP, ci: float = CI, seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """bootstrap resampling estimate of the confidence interval around an auc.

    resamples (y_true, y_score) pairs with replacement n_bootstrap times,
    recomputes auc each time, and returns the percentiles bounding `ci` of
    the resulting distribution. resamples with only one class present can't
    produce an auc and are skipped.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        sample_true = y_true[idx]
        if len(np.unique(sample_true)) < 2:
            continue
        aucs.append(roc_auc_score(sample_true, y_score[idx]))

    lower_pct = (1 - ci) / 2 * 100
    upper_pct = (1 + ci) / 2 * 100
    return float(np.percentile(aucs, lower_pct)), float(np.percentile(aucs, upper_pct))


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
        "combined_l1_l2": "combined_l1_l2_wrongness",
    }

    has_layer3 = (
        "semantic_equivalence_wrongness" in df.columns
        and df["semantic_equivalence_wrongness"].null_count() == 0
    )
    if has_layer3:
        methods["semantic_equivalence"] = "semantic_equivalence_wrongness"
        methods["combined_all_layers"] = "combined_all_layers_wrongness"
    else:
        print("(no layer 3 data yet for all labeled entries -- run `python evaluate.py --layer3` first to include it)")

    results = {"n_labeled": len(entries), "methods": {}}
    print(f"\nevaluated on {len(entries)} labeled entries\n")
    print(f"{'method':22} {'auc':>6}  {'95% ci':>14}")
    for name, column in methods.items():
        scores = df[column].to_list()
        auc = roc_auc_score(wrong, scores)
        lower, upper = bootstrap_auc_ci(wrong, scores)
        results["methods"][name] = {"auc": auc, "ci_low": lower, "ci_high": upper}
        print(f"{name:22} {auc:.3f}  [{lower:.2f}, {upper:.2f}]")

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved results to {results_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layer3", action="store_true",
        help="run layer 3 (paraphrase cross-check) on any dataset entries missing it before evaluating"
    )
    args = parser.parse_args()
    if args.layer3:
        populate_layer3_scores()
    run_evaluation()
