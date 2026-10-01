import csv
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

BASE = Path("data/normalized")
OUT = Path("output/v3_address_recovered.tsv")

S1_FILE = BASE / "train_source1.tsv"
S2_FILE = BASE / "train_source2.tsv"
S3_FILE = BASE / "train_source3.tsv"
GT_FILE = Path("data/train_ground_truth.tsv")

CHUNK = 50_000

print("=" * 70)
print("V3 — STREAMING ADDRESS-TOKEN RECOVERY")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load S1 records
# ------------------------------------------------------------

print("\n[1] Loading S1...")

s1 = {}

for chunk in pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str,
    keep_default_na=False,
    chunksize=CHUNK
):
    for row in chunk.itertuples(index=False):
        s1[row.entity_id] = (
            row.country_norm,
            row.business_name_norm,
            row.business_address_norm
        )

print("S1 loaded:", f"{len(s1):,}")

# ------------------------------------------------------------
# 2. Load S1 address-token index
# ------------------------------------------------------------

print("\n[2] Building S1 address-token index...")

token_index = defaultdict(set)

token_re = re.compile(r"[a-z0-9]+")

for sid, (country, name, address) in s1.items():

    tokens = set(token_re.findall(address.lower()))

    for tok in tokens:
        if len(tok) >= 5:
            key = (country, tok[:5])
            token_index[key].add(sid)

print("Index keys:", f"{len(token_index):,}")

# ------------------------------------------------------------
# 3. Load difficult S1 IDs
# ------------------------------------------------------------

print("\n[3] Loading difficult S1 IDs...")

difficult_file = Path("output/v2_missed_s1.tsv")

difficult_ids = set()

with open(difficult_file, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")

    for row in reader:
        difficult_ids.add(row["source1_entity_id"])

print("Difficult S1:", f"{len(difficult_ids):,}")

# Keep only difficult S1s in the token index
filtered_index = defaultdict(set)

for key, ids in token_index.items():
    selected = ids & difficult_ids

    if selected:
        filtered_index[key] = selected

del token_index

print("Filtered index keys:", f"{len(filtered_index):,}")

# ------------------------------------------------------------
# 4. Stream target sources
# ------------------------------------------------------------

OUT.parent.mkdir(exist_ok=True)

seen = set()
written = 0

out = open(OUT, "w", encoding="utf-8")

out.write(
    "source1_entity_id\tcandidate_entity_id\tsource\n"
)

def process_target_file(path, source_name):

    global written

    print(f"\nProcessing {source_name}: {path}")

    rows_seen = 0
    matches = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK
    ):

        for row in chunk.itertuples(index=False):

            rows_seen += 1

            country = row.country_norm
            address = row.business_address_norm

            tokens = set(token_re.findall(address.lower()))

            candidate_s1 = set()

            for tok in tokens:
                if len(tok) >= 5:
                    key = (country, tok[:5])

                    ids = filtered_index.get(key)

                    if ids:
                        candidate_s1.update(ids)

            if not candidate_s1:
                continue

            target_id = row.entity_id

            for sid in candidate_s1:

                key = (sid, target_id)

                if key in seen:
                    continue

                seen.add(key)

                out.write(
                    f"{sid}\t{target_id}\t{source_name}\n"
                )

                written += 1
                matches += 1

        if rows_seen % 500_000 == 0:
            print(
                f"  scanned={rows_seen:,} "
                f"recovered={matches:,}"
            )

    print(
        f"Finished {source_name}: "
        f"scanned={rows_seen:,}, "
        f"recovered={matches:,}"
    )

# ------------------------------------------------------------
# 5. Run S2 + S3
# ------------------------------------------------------------

process_target_file(S2_FILE, "source2")
process_target_file(S3_FILE, "source3")

out.close()

print()
print("=" * 70)
print("RESULT")
print("=" * 70)
print("Recovered candidate pairs:", f"{written:,}")
print("Output:", OUT)
print("=" * 70)
