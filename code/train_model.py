import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score
)

from lightgbm import LGBMClassifier


BASE = Path(".")
INPUT = BASE / "output" / "train_features.tsv"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "entity_matcher.txt"


print("=" * 70)
print("TRAINING ENTITY MATCHER")
print("=" * 70)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("\n[1/5] Loading training features...")

df = pd.read_csv(
    INPUT,
    sep="\t"
)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns):,}")


# ------------------------------------------------------------
# FEATURES
# ------------------------------------------------------------

target = "label"

X = df.drop(columns=[target])
y = df[target]


# ------------------------------------------------------------
# TRAIN / VALIDATION SPLIT
# ------------------------------------------------------------

print("\n[2/5] Creating train/validation split...")

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training rows:   {len(X_train):,}")
print(f"Validation rows: {len(X_valid):,}")

print("\nClass distribution:")

print(
    y_train.value_counts()
)

# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

print("\n[3/5] Training LightGBM...")

model = LGBMClassifier(
    objective="binary",

    n_estimators=600,
    learning_rate=0.05,

    num_leaves=63,
    max_depth=-1,

    min_child_samples=50,

    subsample=0.85,
    colsample_bytree=0.85,

    reg_alpha=0.1,
    reg_lambda=0.5,

    random_state=42,

    n_jobs=2
)


model.fit(
    X_train,
    y_train
)


# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

print("\n[4/5] Evaluating validation set...")

prob = model.predict_proba(X_valid)[:, 1]

pred = (prob >= 0.5).astype(int)


precision = precision_score(
    y_valid,
    pred,
    zero_division=0
)

recall = recall_score(
    y_valid,
    pred,
    zero_division=0
)

f1 = f1_score(
    y_valid,
    pred,
    zero_division=0
)

average_precision = average_precision_score(
    y_valid,
    prob
)

roc_auc = roc_auc_score(
    y_valid,
    prob
)


print("\nValidation results at threshold 0.50:")
print(f"Precision:          {precision:.6f}")
print(f"Recall:             {recall:.6f}")
print(f"F1:                 {f1:.6f}")
print(f"Average Precision:  {average_precision:.6f}")
print(f"ROC AUC:            {roc_auc:.6f}")


# ------------------------------------------------------------
# FEATURE IMPORTANCE
# ------------------------------------------------------------

print("\nFeature importance:")

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values(
    "importance",
    ascending=False
)

print(
    importance.to_string(index=False)
)


# ------------------------------------------------------------
# SAVE MODEL
# ------------------------------------------------------------

print("\n[5/5] Saving model...")

model.booster_.save_model(
    str(MODEL_PATH)
)

print(f"Model saved: {MODEL_PATH}")


# Save feature order
feature_file = MODEL_DIR / "feature_columns.txt"

with open(feature_file, "w") as f:
    for column in X.columns:
        f.write(column + "\n")

print(f"Feature list saved: {feature_file}")


print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETE")
print("=" * 70)
