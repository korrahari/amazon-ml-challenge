import pandas as pd
import re
import random

MISSED = "output/v2_missed_s1.tsv"

S1_FILE = "data/normalized/train_source1.tsv"
S2_FILE = "data/normalized/train_source2.tsv"
S3_FILE = "data/normalized/train_source3.tsv"

SAMPLE_SIZE = 500

print("=" * 65)
print("V3 SIGNAL TEST")
print("=" * 65)

# --------------------------------------------------
# Load difficult S1s and their actual missed matches
# --------------------------------------------------

missed = pd.read_csv(
    MISSED,
    sep="\t",
    dtype=str
).fillna("")

sample = missed.sample(
    n=min(SAMPLE_SIZE, len(missed)),
    random_state=42
)

print("Sample S1:", len(sample))

# Map S1 -> true target IDs
truth = {}

for r in sample.itertuples(index=False):
    ids = {
        x.strip()
        for x in str(r.matched_entity_ids).split(",")
        if x.strip()
    }

    truth[r.source1_entity_id] = ids

target_ids = set()

for ids in truth.values():
    target_ids.update(ids)

print("Unique true target IDs:", len(target_ids))

# --------------------------------------------------
# Load only relevant target records while streaming
# --------------------------------------------------

target_records = {}

for source, path in [
    ("S2", S2_FILE),
    ("S3", S3_FILE)
]:

    print("Scanning", source)

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=100000,
        usecols=[
            "entity_id",
            "business_name_norm",
            "business_name_ascii",
            "business_name_compact",
            "business_name_token_sorted",
            "business_address_norm",
            "business_address_ascii",
            "business_address_compact",
            "business_address_token_sorted",
            "country_norm",
            "house_number"
        ]
    ):

        chunk = chunk.fillna("")

        wanted = chunk[
            chunk["entity_id"].isin(target_ids)
        ]

        for r in wanted.itertuples(index=False):

            target_records[r.entity_id] = r

print("Target records found:", len(target_records))

# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def chars(x, n):
    x = re.sub(
        r"[^a-z0-9]",
        "",
        str(x).lower()
    )
    return x[:n]

def numbers(x):
    return set(
        re.findall(
            r"\d+",
            str(x)
        )
    )

def address_tokens(x):
    return set(
        re.findall(
            r"[a-z0-9]+",
            str(x).lower()
        )
    )

# --------------------------------------------------
# Load sampled S1 records
# --------------------------------------------------

sample_ids = set(truth.keys())

s1_records = {}

for chunk in pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str,
    chunksize=100000,
    usecols=[
        "entity_id",
        "business_name_norm",
        "business_name_ascii",
        "business_name_compact",
        "business_name_token_sorted",
        "business_address_norm",
        "business_address_ascii",
        "business_address_compact",
        "business_address_token_sorted",
        "country_norm",
        "house_number"
    ]
):

    chunk = chunk.fillna("")

    wanted = chunk[
        chunk["entity_id"].isin(sample_ids)
    ]

    for r in wanted.itertuples(index=False):
        s1_records[r.entity_id] = r

# --------------------------------------------------
# Test signals
# --------------------------------------------------

total_pairs = 0

signals = {
    "country": 0,
    "name4": 0,
    "name5": 0,
    "compact_name": 0,
    "token_name": 0,
    "address5": 0,
    "address_token5": 0,
    "compact_address": 0,
    "token_address": 0,
    "house_number": 0,
    "number_plus_name4": 0,
    "number_plus_address5": 0
}

for sid, true_ids in truth.items():

    s1 = s1_records.get(sid)

    if s1 is None:
        continue

    for tid in true_ids:

        t = target_records.get(tid)

        if t is None:
            continue

        total_pairs += 1

        country_same = (
            s1.country_norm != "" and
            s1.country_norm == t.country_norm
        )

        name4_same = (
            chars(s1.business_name_norm, 4) != "" and
            chars(s1.business_name_norm, 4)
            ==
            chars(t.business_name_norm, 4)
        )

        name5_same = (
            chars(s1.business_name_norm, 5) != "" and
            chars(s1.business_name_norm, 5)
            ==
            chars(t.business_name_norm, 5)
        )

        compact_name_same = (
            s1.business_name_compact != "" and
            s1.business_name_compact
            ==
            t.business_name_compact
        )

        token_name_same = (
            s1.business_name_token_sorted != "" and
            s1.business_name_token_sorted
            ==
            t.business_name_token_sorted
        )

        address5_same = (
            chars(s1.business_address_norm, 5) != "" and
            chars(s1.business_address_norm, 5)
            ==
            chars(t.business_address_norm, 5)
        )

        a1 = address_tokens(s1.business_address_norm)
        a2 = address_tokens(t.business_address_norm)

        address_token5_same = bool(
            {
                x[:5] for x in a1 if len(x) >= 5
            }
            &
            {
                x[:5] for x in a2 if len(x) >= 5
            }
        )

        compact_address_same = (
            s1.business_address_compact != "" and
            s1.business_address_compact
            ==
            t.business_address_compact
        )

        token_address_same = (
            s1.business_address_token_sorted != "" and
            s1.business_address_token_sorted
            ==
            t.business_address_token_sorted
        )

        n1 = numbers(s1.business_address_norm)
        n2 = numbers(t.business_address_norm)

        number_same = bool(n1 & n2)

        number_name4 = (
            number_same and name4_same
        )

        number_address5 = (
            number_same and address5_same
        )

        checks = {
            "country": country_same,
            "name4": name4_same,
            "name5": name5_same,
            "compact_name": compact_name_same,
            "token_name": token_name_same,
            "address5": address5_same,
            "address_token5": address_token5_same,
            "compact_address": compact_address_same,
            "token_address": token_address_same,
            "house_number": number_same,
            "number_plus_name4": number_name4,
            "number_plus_address5": number_address5
        }

        for key, value in checks.items():
            if value:
                signals[key] += 1

print()
print("=" * 65)
print("RESULT")
print("=" * 65)

print("True missed pairs tested:", total_pairs)
print()

for key, value in signals.items():

    recall = (
        value / total_pairs * 100
        if total_pairs else 0
    )

    print(
        f"{key:25s}"
        f"{value:7d}"
        f"  {recall:7.2f}%"
    )

print("=" * 65)
