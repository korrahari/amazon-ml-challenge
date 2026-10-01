import pandas as pd
import sqlite3
from collections import Counter

GT = "data/train_ground_truth.tsv"
S1 = "data/normalized/train_source1.tsv"
S2 = "data/normalized/train_source2.tsv"
S3 = "data/normalized/train_source3.tsv"
DB = "output/v3_index.sqlite"

print("Loading existing V3 predictions...")

pred = pd.read_csv(
    "output/train_predicted_v3.tsv",
    sep="\t",
    dtype=str,
    usecols=["source1_entity_id", "candidate_entity_id"]
)

pred_pairs = set(
    zip(pred.source1_entity_id, pred.candidate_entity_id)
)

print("Existing prediction pairs:", f"{len(pred_pairs):,}")

print("Loading ground truth...")

gt = pd.read_csv(
    GT,
    sep="\t",
    dtype=str,
    keep_default_na=False
)

remaining_targets = set()
remaining_pairs = []

for r in gt.itertuples(index=False):

    ids = str(r.matched_entity_ids).strip()

    if not ids:
        continue

    for target in ids.split(","):

        target = target.strip()

        if not target:
            continue

        pair = (r.source1_entity_id, target)

        if pair not in pred_pairs:
            remaining_pairs.append(pair)
            remaining_targets.add(target)

print("Remaining true pairs:", f"{len(remaining_pairs):,}")
print("Unique remaining target IDs:", f"{len(remaining_targets):,}")

print("\nLoading target key frequencies...")

conn = sqlite3.connect(DB)

freq = {}

for country, token5, frequency in conn.execute("""
    SELECT country, token5, frequency
    FROM token_frequency
"""):
    freq[(country, token5)] = int(frequency)

conn.close()

print("Frequency keys loaded:", f"{len(freq):,}")

# We need target_id -> (country, token5)
target_keys = {}

for source, path in [
    ("S2", S2),
    ("S3", S3)
]:

    print("Scanning:", source)

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "entity_id",
            "country_norm",
            "business_address_token_sorted"
        ],
        chunksize=100000
    ):

        hit = chunk[
            chunk.entity_id.isin(remaining_targets)
        ]

        for r in hit.itertuples(index=False):

            key = (
                r.country_norm,
                r.business_address_token_sorted[:5]
            )

            target_keys[r.entity_id] = key

print("\nTarget keys found:", f"{len(target_keys):,}")

bands = Counter()

for s1_id, target_id in remaining_pairs:

    key = target_keys.get(target_id)

    if key is None:
        bands["MISSING"] += 1
        continue

    f = freq.get(key, 0)

    if f <= 5:
        bands["1-5"] += 1
    elif f <= 10:
        bands["6-10"] += 1
    elif f <= 25:
        bands["11-25"] += 1
    elif f <= 50:
        bands["26-50"] += 1
    elif f <= 100:
        bands["51-100"] += 1
    elif f <= 250:
        bands["101-250"] += 1
    elif f <= 500:
        bands["251-500"] += 1
    elif f <= 1000:
        bands["501-1000"] += 1
    elif f <= 5000:
        bands["1001-5000"] += 1
    else:
        bands["5000+"] += 1

print()
print("=" * 70)
print("V3 REMAINING-MISS FREQUENCY ANALYSIS")
print("=" * 70)

total = len(remaining_pairs)

for band in [
    "1-5",
    "6-10",
    "11-25",
    "26-50",
    "51-100",
    "101-250",
    "251-500",
    "501-1000",
    "1001-5000",
    "5000+",
    "MISSING"
]:

    n = bands[band]

    print(
        f"{band:>10}: "
        f"{n:>10,} "
        f"({100*n/total:6.2f}%)"
    )

print("=" * 70)
