import pandas as pd
import os
PRED = "output/test_predicted_matches.tsv"
OUT = "output/matching_results_v2.tsv"

print("Loading combined predictions...")

pred = pd.read_csv(
    PRED,
    sep="\t",
    dtype=str
).fillna("")

# Keep predictions at the trained threshold
pred["match_probability"] = pd.to_numeric(
    pred["match_probability"],
    errors="coerce"
)

pred = pred[pred["match_probability"] >= 0.78].copy()

print("Prediction rows:", len(pred))

# Group matched IDs for each S1
grouped = (
    pred.groupby("source1_entity_id")["candidate_entity_id"]
    .apply(lambda x: ",".join(dict.fromkeys(x)))
)

# Load ALL test S1 IDs so every S1 appears in submission.
test_s1 = pd.read_csv(
    "data/normalized/test_source1.tsv",
    sep="\t",
    dtype=str,
    usecols=["entity_id"]
)

result = pd.DataFrame({
    "source1_entity_id": test_s1["entity_id"]
})

result["matched_entity_ids"] = (
    result["source1_entity_id"]
    .map(grouped)
    .fillna("")
)

# Exact required two-column TSV
result.to_csv(
    OUT,
    sep="\t",
    index=False
)

print()
print("V2 CREATED")
print("Rows:", len(result))
print("Rows with matches:", (result["matched_entity_ids"] != "").sum())
print("Rows without matches:", (result["matched_entity_ids"] == "").sum())
print("Output:", OUT)
print()
print("V1 remains untouched:")
print("output/matching_results.tsv")
