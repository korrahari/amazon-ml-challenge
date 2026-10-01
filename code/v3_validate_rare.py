import pandas as pd
from collections import defaultdict

PRED = "output/v3_rare_predicted_matches.tsv"
GT = "data/train_ground_truth.tsv"

print("Loading predictions...")
pred = pd.read_csv(PRED, sep="\t", dtype=str)

print("Predictions:", len(pred))

print("Loading ground truth...")
gt = pd.read_csv(GT, sep="\t", dtype=str, keep_default_na=False)

truth = defaultdict(set)

for row in gt.itertuples(index=False):
    ids = str(row.matched_entity_ids).strip()

    if ids:
        for x in ids.split(","):
            x = x.strip()
            if x:
                truth[row.source1_entity_id].add(x)

tp = 0
fp = 0

for row in pred.itertuples(index=False):

    if row.candidate_entity_id in truth.get(
        row.source1_entity_id, set()
    ):
        tp += 1
    else:
        fp += 1

precision = tp / len(pred) if len(pred) else 0

print()
print("=" * 60)
print("V3 RARE PREDICTION VALIDATION")
print("=" * 60)
print("Predicted:", f"{len(pred):,}")
print("True positives:", f"{tp:,}")
print("False positives:", f"{fp:,}")
print("Precision:", f"{precision:.6f}")
print("=" * 60)
