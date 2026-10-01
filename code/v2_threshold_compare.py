import pandas as pd
import numpy as np

GT = "data/train_ground_truth.tsv"
PRED = "output/train_predicted_combined.tsv"

print("Loading ground truth...")
gt = pd.read_csv(GT, sep="\t", dtype=str).fillna("")

print("Loading predictions...")
pred = pd.read_csv(PRED, sep="\t", dtype=str).fillna("")

# --------------------------------------------------
# Ground truth
# --------------------------------------------------

truth = {}

for r in gt.itertuples(index=False):
    ids = set()

    value = r.matched_entity_ids

    if value:
        ids = {
            x.strip()
            for x in str(value).split(",")
            if x.strip()
        }

    truth[r.source1_entity_id] = ids

# --------------------------------------------------
# Predictions
# Keep only predictions at the trained threshold.
# --------------------------------------------------

pred["match_probability"] = pd.to_numeric(
    pred["match_probability"],
    errors="coerce"
)

pred = pred[pred["match_probability"] >= 0.78]

# Build prediction sets
prediction = {}

for r in pred.itertuples(index=False):
    prediction.setdefault(
        r.source1_entity_id,
        set()
    ).add(r.candidate_entity_id)

# --------------------------------------------------
# Macro F0.5
# --------------------------------------------------

scores = []

for sid, true_ids in truth.items():

    predicted_ids = prediction.get(sid, set())

    tp = len(true_ids & predicted_ids)
    fp = len(predicted_ids - true_ids)
    fn = len(true_ids - predicted_ids)

    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)

    if precision == 0 and recall == 0:
        f05 = 0.0
    else:
        f05 = (
            1.25 * precision * recall
            / (0.25 * precision + recall)
        )

    scores.append(f05)

macro_f05 = np.mean(scores)

print()
print("=" * 50)
print("V2 THRESHOLD EVALUATION")
print("=" * 50)
print("Threshold:       0.78")
print("S1 evaluated:    ", len(scores))
print("Prediction rows: ", len(pred))
print("Macro F0.5:      ", round(macro_f05, 6))
print("Macro F0.5 %:    ", round(macro_f05 * 100, 2))
print("=" * 50)
