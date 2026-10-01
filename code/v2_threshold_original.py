import pandas as pd
import numpy as np

GT = "data/train_ground_truth.tsv"
PRED = "output/train_predicted_matches.tsv"

print("Loading ground truth...")
gt = pd.read_csv(GT, sep="\t", dtype=str).fillna("")

print("Loading original predictions...")
pred = pd.read_csv(PRED, sep="\t", dtype=str).fillna("")

pred["match_probability"] = pd.to_numeric(
    pred["match_probability"],
    errors="coerce"
)

# Ground truth sets
truth = {}

for r in gt.itertuples(index=False):
    value = r.matched_entity_ids

    if value:
        truth[r.source1_entity_id] = {
            x.strip()
            for x in str(value).split(",")
            if x.strip()
        }
    else:
        truth[r.source1_entity_id] = set()

print("Ground truth S1:", len(truth))
print("Original prediction rows:", len(pred))

thresholds = [
    0.70,
    0.72,
    0.74,
    0.76,
    0.78,
    0.80,
    0.82,
    0.84,
    0.86,
    0.88,
    0.90
]

print()
print("=" * 65)
print("THRESHOLD       MACRO F0.5")
print("=" * 65)

for threshold in thresholds:

    p = pred[
        pred["match_probability"] >= threshold
    ]

    prediction = {}

    for r in p.itertuples(index=False):
        prediction.setdefault(
            r.source1_entity_id,
            set()
        ).add(r.candidate_entity_id)

    scores = []

    for sid, true_ids in truth.items():

        predicted_ids = prediction.get(sid, set())

        tp = len(true_ids & predicted_ids)
        fp = len(predicted_ids - true_ids)
        fn = len(true_ids - predicted_ids)

        precision = (
            tp / (tp + fp)
            if tp + fp else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn else 0.0
        )

        if precision == 0 and recall == 0:
            f05 = 0.0
        else:
            f05 = (
                1.25 * precision * recall
                / (0.25 * precision + recall)
            )

        scores.append(f05)

    macro = np.mean(scores)

    print(
        f"{threshold:<15.2f}"
        f"{macro:.6f}"
        f"  ({macro * 100:.2f}%)"
    )

print("=" * 65)
