import json

DATASET_PATH = "dataset.jsonl"


def load_entries(path: str = DATASET_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_entries(entries: list[dict], path: str = DATASET_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def label_dataset(path: str = DATASET_PATH) -> None:
    entries = load_entries(path)
    unlabeled = [e for e in entries if e.get("label") is None]

    if not unlabeled:
        print("nothing left to label.")
        return

    print(f"{len(unlabeled)} unlabeled entries.")
    print("1 = semantically correct, 0 = semantically wrong, s = skip, q = save & quit\n")

    for i, entry in enumerate(unlabeled, start=1):
        print(f"--- {i}/{len(unlabeled)} ---")
        print(f"question: {entry['question']}")
        print(f"query:\n{entry['generated_query']}\n")

        while True:
            answer = input("label (1/0/s/q): ").strip().lower()
            if answer in ("1", "0"):
                entry["label"] = int(answer)
                break
            elif answer == "s":
                break
            elif answer == "q":
                save_entries(entries, path)
                print(f"\nsaved progress to {path}")
                return
            else:
                print("enter 1, 0, s, or q")
        print()

    save_entries(entries, path)
    print(f"saved progress to {path}")


if __name__ == "__main__":
    label_dataset()
