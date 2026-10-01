import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, fbeta_score
from lightgbm import Booster

BASE = Path(".")
INPUT = BASE / "output" / "train_features.tsv"
MODEL_PATH = BASE / "models" / "entity_matcher.txt"

print("=" * 70)
print("TUNING F0.5 THRESHOLD")
print("=" * 70)

print("\n[1/4] Loading validation data...")

df = pd.read_csv(INPUT, sep="\t")

feature_columns = [
    c for c in df.columns
    if c != "label"
]

X = df[feature_columns]
y = df["label"]

# Reproduce the same 80/20 split used during training
from sklearn.model_selection import train_test_split

_, X_valid, _, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Validation rows: {len(X_valid):,}")

print("\n[2/4] Loading trained model...")

model = Booster(
    model_file=str(MODEL_PATH)
)

print("\n[3/4] Generating probabilities...")

probabilities = model.predict(X_valid)

print("Probabilities generated.")

print("\n[4/4] Testing thresholds...")

thresholds = np.arange(
    0.30,
    0.991,
    0.01
)

results = []

for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_valid,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_valid,
        predictions,
        zero_division=0
    )

    f05 = fbeta_score(
        y_valid,
        predictions,
        beta=0.5,
        zero_division=0
    )

    results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f0.5": f05
    })


results_df = pd.DataFrame(results)

best = results_df.loc[
    results_df["f0.5"].idxmax()
]

print("\n" + "=" * 70)
print("F0.5 THRESHOLD RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        formatters={
            "threshold": "{:.2f}".format,
            "precision": "{:.6f}".format,
            "recall": "{:.6f}".format,
            "f0.5": "{:.6f}".format
        }
    )
)

print("\n" + "=" * 70)
print("BEST VALIDATION THRESHOLD")
print("=" * 70)

print(f"Threshold: {best['threshold']:.2f}")
print(f"Precision: {best['precision']:.6f}")
print(f"Recall:    {best['recall']:.6f}")
print(f"F0.5:      {best['f0.5']:.6f}")

threshold_file = BASE / "models" / "best_threshold.txt"

with open(threshold_file, "w") as f:
    f.write(f"{best['threshold']:.2f}\n")

print(f"\nSaved threshold: {threshold_file}")

print("\n" + "=" * 70)
print("THRESHOLD TUNING COMPLETE")
print("=" * 70)
