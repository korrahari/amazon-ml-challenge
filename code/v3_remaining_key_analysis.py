import pandas as pd
from collections import Counter

GT = "data/train_ground_truth.tsv"
S1 = "data/normalized/train_source1.tsv"
S2 = "data/normalized/train_source2.tsv"
S3 = "data/normalized/train_source3.tsv"

print("Loading ground truth...")
gt = pd.read_csv(GT, sep="\t", dtype=str, keep_default_na=False)

# Build true-pair lookup
truth = {}

for r in gt.itertuples(index=False):
    ids = str(r.matched_entity_ids).strip()
    truth[r.source1_entity_id] = [
        x.strip() for x in ids.split(",") if x.strip()
    ]

print("Loading S1...")
s1 = pd.read_csv(
    S1,
    sep="\t",
    dtype=str,
    keep_default_na=False,
    usecols=[
        "entity_id",
        "country_norm",
        "business_address_token_sorted"
    ]
)

s1 = s1.set_index("entity_id")

# Existing V3 predictions
pred = pd.read_csv(
    "output/train_predicted_v3.tsv",
    sep="\t",
    dtype=str
)

pred_pairs = set(
    zip(
        pred.source1_entity_id,
        pred.candidate_entity_id
    )
)

print("Existing predictions:", len(pred_pairs))

# Count remaining true pairs by S1 address-token key
remaining_keys = Counter()
remaining_pairs = 0

for s1_id, targets in truth.items():

    if s1_id not in s1.index:
        continue

    row = s1.loc[s1_id]

    key = (
        row.country_norm,
        row.business_address_token_sorted[:5]
    )

    for target in targets:

        if (s1_id, target) not in pred_pairs:
            remaining_keys[key] += 1
            remaining_pairs += 1

print()
print("=" * 70)
print("V3 REMAINING-MISS KEY ANALYSIS")
print("=" * 70)
print("Remaining true pairs:", f"{remaining_pairs:,}")
print("Unique address-token keys:", f"{len(remaining_keys):,}")

print()
print("Top 30 keys by remaining true pairs:")

for key, count in remaining_keys.most_common(30):
    print(
        f"{count:6d} | "
        f"country={key[0]} | "
        f"token5={key[1]!r}"
    )

print("=" * 70)
