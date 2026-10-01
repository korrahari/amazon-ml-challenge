import csv
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

BASE = Path("data/normalized")
OUT = Path("output/v3_recovered_matches.tsv")

S1_FILE = BASE / "train_source1.tsv"
S2_FILE = BASE / "train_source2.tsv"
S3_FILE = BASE / "train_source3.tsv"
DIFFICULT = Path("output/v2_missed_s1.tsv")

CHUNK = 50_000

# Conservative thresholds.
# We only recover when address-token overlap is supported
# by strong name/address evidence.
NAME_STRONG = 88
NAME_MEDIUM = 78
ADDR_STRONG = 65
ADDR_MEDIUM = 52

token_re = re.compile(r"[a-z0-9]+")


def tokens5(text):
    return {
        x[:5]
        for x in token_re.findall(str(text).lower())
        if len(x) >= 5
    }


print("=" * 70)
print("V3 — SCORED ADDRESS-TOKEN RECOVERY")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load difficult S1 IDs
# ------------------------------------------------------------

print("\n[1] Loading difficult S1 IDs...")

difficult_ids = set()

with open(DIFFICULT, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")

    for row in reader:
        difficult_ids.add(row["source1_entity_id"])

print("Difficult S1:", f"{len(difficult_ids):,}")

# ------------------------------------------------------------
# 2. Load only difficult S1 records
# ------------------------------------------------------------

print("\n[2] Loading difficult S1 records...")

s1 = {}

for chunk in pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str,
    keep_default_na=False,
    chunksize=CHUNK
):
    for row in chunk.itertuples(index=False):

        sid = row.entity_id

        if sid in difficult_ids:
            s1[sid] = {
                "country": row.country_norm,
                "name": row.business_name_norm,
                "address": row.business_address_norm,
                "tokens": tokens5(row.business_address_norm)
            }

print("Loaded difficult S1 records:", f"{len(s1):,}")

# ------------------------------------------------------------
# 3. Build compact index
# ------------------------------------------------------------

print("\n[3] Building compact address-token index...")

index = defaultdict(list)

for sid, r in s1.items():

    for tok in r["tokens"]:

        index[(r["country"], tok)].append(sid)

print("Index keys:", f"{len(index):,}")

# ------------------------------------------------------------
# 4. Stream target files
# ------------------------------------------------------------

OUT.parent.mkdir(exist_ok=True)

# Keep only best score for each S1-target pair.
best = {}

def process_target_file(path, source):

    print()
    print("Processing:", source)

    scanned = 0
    shared = 0
    accepted = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK
    ):

        for row in chunk.itertuples(index=False):

            scanned += 1

            country = row.country_norm
            target_name = row.business_name_norm
            target_address = row.business_address_norm

            target_tokens = tokens5(target_address)

            if not target_tokens:
                continue

            candidate_s1 = set()

            for tok in target_tokens:

                ids = index.get((country, tok))

                if ids:
                    candidate_s1.update(ids)

            if not candidate_s1:
                continue

            shared += len(candidate_s1)

            for sid in candidate_s1:

                r = s1[sid]

                name_score = fuzz.WRatio(
                    r["name"],
                    target_name
                )

                addr_score = fuzz.WRatio(
                    r["address"],
                    target_address
                )

                shared_tokens = len(
                    r["tokens"] & target_tokens
                )

                # Conservative acceptance rules.
                accept = False

                if shared_tokens >= 2:
                    if name_score >= NAME_MEDIUM and addr_score >= ADDR_MEDIUM:
                        accept = True

                elif shared_tokens == 1:

                    if (
                        name_score >= NAME_STRONG
                        and addr_score >= ADDR_MEDIUM
                    ):
                        accept = True

                    elif (
                        name_score >= NAME_MEDIUM
                        and addr_score >= ADDR_STRONG
                    ):
                        accept = True

                if not accept:
                    continue

                target_id = row.entity_id

                key = (sid, target_id, source)

                score = (
                    0.50 * name_score +
                    0.35 * addr_score +
                    15 * min(shared_tokens, 2)
                )

                old = best.get(key)

                if old is None or score > old:
                    best[key] = score
                    accepted += 1

        if scanned % 500_000 == 0:
            print(
                f"  scanned={scanned:,} "
                f"shared={shared:,} "
                f"accepted={accepted:,}"
            )

    print(
        f"Finished {source}: "
        f"scanned={scanned:,}, "
        f"accepted={accepted:,}"
    )


process_target_file(S2_FILE, "source2")
process_target_file(S3_FILE, "source3")

# ------------------------------------------------------------
# 5. Write only accepted matches
# ------------------------------------------------------------

print("\n[5] Writing accepted matches...")

with open(OUT, "w", encoding="utf-8") as f:

    f.write(
        "source1_entity_id\tcandidate_entity_id\tsource\tconfidence\n"
    )

    for (sid, target_id, source), score in best.items():

        f.write(
            f"{sid}\t{target_id}\t{source}\t{score:.2f}\n"
        )

print()
print("=" * 70)
print("RESULT")
print("=" * 70)
print("Accepted matches:", f"{len(best):,}")
print("Output:", OUT)
print("=" * 70)
