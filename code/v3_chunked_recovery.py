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
OUT = Path("output/v3_chunked_recovered.tsv")

CHUNK = 25_000

token_re = re.compile(r"[a-z0-9]+")


def tokens5(text):
    return {
        x[:5]
        for x in token_re.findall(str(text).lower())
        if len(x) >= 5
    }


def compact(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


print("=" * 70)
print("V3 — CHUNKED RECOVERY")
print("=" * 70)

# ------------------------------------------------------------
# 1. Difficult S1 IDs
# ------------------------------------------------------------

print("\n[1] Loading difficult S1 IDs...")

difficult = set()

with open(DIFFICULT_FILE, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")

    for row in reader:
        difficult.add(row["source1_entity_id"])

print("Difficult S1:", f"{len(difficult):,}")

# ------------------------------------------------------------
# 2. Load difficult S1
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

            s1[r.entity_id] = (
                r.country_norm,
                r.business_name_norm,
                compact(r.business_name_norm),
                r.business_address_norm,
                tokens5(r.business_address_norm)
            )

print("Loaded:", f"{len(s1):,}")

# ------------------------------------------------------------
# 3. Build compact index
# ------------------------------------------------------------

print("\n[3] Building S1 address-token index...")

index = defaultdict(set)

for sid, (country, name, name_compact, address, toks) in s1.items():

    for tok in toks:
        index[(country, tok)].add(sid)

print("Index keys:", f"{len(index):,}")

# ------------------------------------------------------------
# 4. Stream one target source
# ------------------------------------------------------------

accepted = {}

def process_target_file(path, source):

    print()
    print("=" * 70)
    print("PROCESSING", source)
    print("=" * 70)

    scanned = 0
    candidate_refs = 0
    screened = 0
    accepted_now = 0

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
            target_name = r.business_name_norm
            target_name_compact = compact(target_name)
            target_address = r.business_address_norm

            target_tokens = tokens5(target_address)

            if not target_tokens:
                continue

            # Retrieve only S1s sharing an address token.
            candidate_ids = set()

            for tok in target_tokens:

                ids = index.get((country, tok))

                if ids:
                    candidate_ids.update(ids)

            if not candidate_ids:
                continue

            candidate_refs += len(candidate_ids)

            for sid in candidate_ids:

                s_country, s_name, s_name_compact, s_address, s_tokens = s1[sid]

                shared = len(s_tokens & target_tokens)

                # Cheap filters.
                if shared >= 3:
                    pass

                elif shared == 2:
                    if (
                        s_name_compact != target_name_compact
                        and
                        abs(
                            len(s_name_compact)
                            - len(target_name_compact)
                        ) > 15
                    ):
                        continue

                else:
                    # One shared token requires strong name similarity.
                    if (
                        s_name_compact != target_name_compact
                        and
                        abs(
                            len(s_name_compact)
                            - len(target_name_compact)
                        ) > 10
                    ):
                        continue

                screened += 1

                # Fuzzy only after cheap screening.
                name_score = fuzz.WRatio(
                    s_name,
                    target_name
                )

                # Fast address scorer.
                address_score = fuzz.ratio(
                    s_address,
                    target_address
                )

                accept = False

                if shared >= 3:

                    if (
                        name_score >= 70
                        and
                        address_score >= 40
                    ):
                        accept = True

                elif shared == 2:

                    if (
                        name_score >= 82
                        and
                        address_score >= 50
                    ):
                        accept = True

                else:

                    if (
                        name_score >= 90
                        and
                        address_score >= 55
                    ):
                        accept = True

                if not accept:
                    continue

                target_id = r.entity_id

                confidence = (
                    0.55 * name_score
                    + 0.35 * address_score
                    + 5.0 * min(shared, 2)
                )

                key = (
                    sid,
                    target_id,
                    source
                )

                old = accepted.get(key)

                if old is None or confidence > old:
                    accepted[key] = confidence
                    accepted_now += 1

        if scanned % 250_000 == 0:

            print(
                f"scanned={scanned:,} "
                f"candidate_refs={candidate_refs:,} "
                f"screened={screened:,} "
                f"accepted={accepted_now:,}"
            )

    print()
    print("Finished:", source)
    print("Scanned:", f"{scanned:,}")
    print("Candidate refs:", f"{candidate_refs:,}")
    print("Screened:", f"{screened:,}")
    print("Accepted:", f"{accepted_now:,}")


# ------------------------------------------------------------
# 5. Process both sources
# ------------------------------------------------------------

process_target_file(
    S2_FILE,
    "source2"
)

process_target_file(
    S3_FILE,
    "source3"
)

# ------------------------------------------------------------
# 6. Write only accepted matches
# ------------------------------------------------------------

print("\n[6] Writing accepted matches...")

OUT.parent.mkdir(exist_ok=True)

with open(
    OUT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "source1_entity_id\t"
        "candidate_entity_id\t"
        "source\t"
        "confidence\n"
    )

    for (
        sid,
        target_id,
        source
    ), confidence in accepted.items():

        f.write(
            f"{sid}\t"
            f"{target_id}\t"
            f"{source}\t"
            f"{confidence:.2f}\n"
        )

print()
print("=" * 70)
print("V3 CHUNKED RECOVERY COMPLETE")
print("=" * 70)
print(
    "Accepted matches:",
    f"{len(accepted):,}"
)
print("Output:", OUT)
print("=" * 70)
