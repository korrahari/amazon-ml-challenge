import csv
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

BASE = Path("data/normalized")
S1_FILE = BASE / "train_source1.tsv"
S2_FILE = BASE / "train_source2.tsv"
S3_FILE = BASE / "train_source3.tsv"
DIFFICULT_FILE = Path("output/v2_missed_s1.tsv")
OUT = Path("output/v3_fast_recovered.tsv")

CHUNK = 50_000

token_re = re.compile(r"[a-z0-9]+")


def addr_tokens(text):
    return {
        x[:5]
        for x in token_re.findall(str(text).lower())
        if len(x) >= 5
    }


def compact(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


print("=" * 70)
print("V3 — FAST RECOVERY")
print("=" * 70)

# ------------------------------------------------------------
# 1. Difficult S1 IDs
# ------------------------------------------------------------

print("\n[1] Loading difficult S1 IDs...")

difficult = set()

with open(DIFFICULT_FILE, encoding="utf-8") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        difficult.add(row["source1_entity_id"])

print("Difficult S1:", f"{len(difficult):,}")

# ------------------------------------------------------------
# 2. Load difficult S1 records
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
    for r in chunk.itertuples(index=False):

        if r.entity_id in difficult:
            s1[r.entity_id] = {
                "country": r.country_norm,
                "name": r.business_name_norm,
                "name_compact": compact(r.business_name_norm),
                "address": r.business_address_norm,
                "tokens": addr_tokens(r.business_address_norm),
            }

print("Loaded:", f"{len(s1):,}")

# ------------------------------------------------------------
# 3. Build compact index
# ------------------------------------------------------------

print("\n[3] Building address-token index...")

index = defaultdict(list)

for sid, r in s1.items():
    for token in r["tokens"]:
        index[(r["country"], token)].append(sid)

print("Index keys:", f"{len(index):,}")

# ------------------------------------------------------------
# 4. Stream targets
# ------------------------------------------------------------

accepted = {}

def process(path, source):

    print()
    print("=" * 60)
    print("SOURCE:", source)
    print("=" * 60)

    scanned = 0
    candidate_count = 0
    screened = 0
    accepted_count = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK
    ):

        for r in chunk.itertuples(index=False):

            scanned += 1

            country = r.country_norm
            tname = r.business_name_norm
            tname_compact = compact(tname)
            taddr = r.business_address_norm

            ttokens = addr_tokens(taddr)

            if not ttokens:
                continue

            candidates = set()

            for token in ttokens:
                ids = index.get((country, token))

                if ids:
                    candidates.update(ids)

            if not candidates:
                continue

            candidate_count += len(candidates)

            for sid in candidates:

                sr = s1[sid]

                shared = len(sr["tokens"] & ttokens)

                if shared == 0:
                    continue

                # Cheap screening BEFORE expensive fuzzy scoring.
                name_compact_same = (
                    sr["name_compact"] == tname_compact
                )

                name_len_diff = abs(
                    len(sr["name_compact"])
                    - len(tname_compact)
                )

                # Strong name evidence.
                if not name_compact_same and name_len_diff > 20:

                    # Still allow candidates with 2+ address tokens.
                    if shared < 2:
                        continue

                screened += 1

                # Fuzzy scoring only after screening.
                name_score = fuzz.WRatio(
                    sr["name"],
                    tname
                )

                addr_score = fuzz.WRatio(
                    sr["address"],
                    taddr
                )

                # Conservative acceptance.
                accept = False

                if shared >= 3:
                    if name_score >= 72 and addr_score >= 45:
                        accept = True

                elif shared == 2:
                    if name_score >= 80 and addr_score >= 50:
                        accept = True

                else:  # one shared address token
                    if name_score >= 90 and addr_score >= 55:
                        accept = True

                if not accept:
                    continue

                target_id = r.entity_id

                # Combined confidence for later threshold tuning.
                confidence = (
                    0.50 * name_score +
                    0.35 * addr_score +
                    5.0 * min(shared, 3)
                )

                key = (sid, target_id, source)

                old = accepted.get(key)

                if old is None or confidence > old:
                    accepted[key] = confidence
                    accepted_count += 1

        if scanned % 500_000 == 0:
            print(
                f"scanned={scanned:,} "
                f"candidates={candidate_count:,} "
                f"screened={screened:,} "
                f"accepted={accepted_count:,}"
            )

    print()
    print("Finished", source)
    print("Scanned:", f"{scanned:,}")
    print("Candidate references:", f"{candidate_count:,}")
    print("Screened:", f"{screened:,}")
    print("Accepted:", f"{accepted_count:,}")


process(S2_FILE, "source2")
process(S3_FILE, "source3")

# ------------------------------------------------------------
# 5. Write accepted matches only
# ------------------------------------------------------------

print("\n[5] Writing results...")

OUT.parent.mkdir(exist_ok=True)

with open(OUT, "w", encoding="utf-8") as f:

    f.write(
        "source1_entity_id\tcandidate_entity_id\tsource\tconfidence\n"
    )

    for (sid, target_id, source), confidence in accepted.items():

        f.write(
            f"{sid}\t{target_id}\t{source}\t{confidence:.2f}\n"
        )

print()
print("=" * 70)
print("V3 FAST RECOVERY COMPLETE")
print("=" * 70)
print("Accepted matches:", f"{len(accepted):,}")
print("Output:", OUT)
print("=" * 70)
