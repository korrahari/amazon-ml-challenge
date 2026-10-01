import pandas as pd
import re
import time

S1_FILE = "data/normalized/train_source1.tsv"
S2_FILE = "data/normalized/train_source2.tsv"
S3_FILE = "data/normalized/train_source3.tsv"
GT_FILE = "data/train_ground_truth.tsv"

print("Loading ground truth...")

gt = pd.read_csv(GT_FILE, sep="\t", dtype=str).fillna("")

# Only S1s whose true matches were missed by V2 baseline
missed = pd.read_csv(
    "output/v2_missed_s1.tsv",
    sep="\t",
    dtype=str
).fillna("")

missed_ids = set(missed["source1_entity_id"])

print("Difficult S1:", len(missed_ids))

# Load only the S1 records we need
s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str,
    usecols=[
        "entity_id",
        "business_name_norm",
        "business_address_norm",
        "country_norm",
        "house_number"
    ]
).fillna("")

s1 = s1[s1.entity_id.isin(missed_ids)].copy()

# Ground truth lookup
gt = gt[gt["source1_entity_id"].isin(missed_ids)]

truth = {}

for r in gt.itertuples(index=False):
    sid = r.source1_entity_id

    # Ground truth uses comma-separated matched S2/S3 IDs
    value = getattr(r, "matched_entity_ids", "")

    if pd.isna(value):
        value = ""

    ids = {
        x.strip()
        for x in str(value).split(",")
        if x.strip()
    }

    truth[sid] = ids
print("S1 with ground truth:", len(truth))

# ---------------------------------------------------------
# Build key functions
# ---------------------------------------------------------

def tokens(x):
    return re.findall(r"[a-z0-9]+", str(x).lower())

def nums(x):
    return set(re.findall(r"\d+", str(x)))

def first_chars(x, n):
    x = re.sub(r"[^a-z0-9]", "", str(x).lower())
    return x[:n]

# ---------------------------------------------------------
# Build target key indexes
# ---------------------------------------------------------

indexes = {
    "name4": {},
    "name5": {},
    "addr_token5": {},
    "addr_token6": {},
    "number": {},
    "number_name4": {},
    "number_addr5": {},
}

def add(index, key, tid):
    if key:
        index.setdefault(key, set()).add(tid)

print("\nBuilding target indexes...")

for source, path in [
    ("S2", S2_FILE),
    ("S3", S3_FILE)
]:
    print("Reading", source)

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=100000,
        usecols=[
            "entity_id",
            "business_name_norm",
            "business_address_norm",
            "country_norm"
        ]
    ):
        chunk = chunk.fillna("")

        for r in chunk.itertuples(index=False):

            tid = r.entity_id
            country = r.country_norm

            name_key = first_chars(r.business_name_norm, 5)

            address_tokens = tokens(r.business_address_norm)
            numbers = nums(r.business_address_norm)

            # Name prefix
            add(
                indexes["name4"],
                (country, first_chars(r.business_name_norm, 4)),
                tid
            )

            add(
                indexes["name5"],
                (country, name_key),
                tid
            )

            # Address tokens
            for t in address_tokens:
                if len(t) >= 5:
                    add(
                        indexes["addr_token5"],
                        (country, t[:5]),
                        tid
                    )

                if len(t) >= 6:
                    add(
                        indexes["addr_token6"],
                        (country, t[:6]),
                        tid
                    )

            # Numbers
            for n in numbers:
                add(
                    indexes["number"],
                    (country, n),
                    tid
                )

                add(
                    indexes["number_name4"],
                    (
                        country,
                        n,
                        first_chars(r.business_name_norm, 4)
                    ),
                    tid
                )

                add(
                    indexes["number_addr5"],
                    (
                        country,
                        n,
                        first_chars(r.business_address_norm, 5)
                    ),
                    tid
                )

print("\nIndexes built.")

# ---------------------------------------------------------
# Test recall using actual missed ground truth
# ---------------------------------------------------------

recovered = {name: 0 for name in indexes}
total_true = 0

for r in s1.itertuples(index=False):

    sid = r.entity_id
    true_ids = truth.get(sid, set())

    if not true_ids:
        continue

    country = r.country_norm

    total_true += len(true_ids)

    name4 = first_chars(r.business_name_norm, 4)
    name5 = first_chars(r.business_name_norm, 5)

    address_tokens = tokens(r.business_address_norm)
    numbers = nums(r.business_address_norm)

    candidates = {}

    # name4
    c = indexes["name4"].get((country, name4), set())
    candidates["name4"] = c

    # name5
    candidates["name5"] = indexes["name5"].get(
        (country, name5), set()
    )

    # address token 5
    c = set()
    for t in address_tokens:
        if len(t) >= 5:
            c.update(
                indexes["addr_token5"].get(
                    (country, t[:5]), set()
                )
            )
    candidates["addr_token5"] = c

    # address token 6
    c = set()
    for t in address_tokens:
        if len(t) >= 6:
            c.update(
                indexes["addr_token6"].get(
                    (country, t[:6]), set()
                )
            )
    candidates["addr_token6"] = c

    # number
    c = set()
    for n in numbers:
        c.update(
            indexes["number"].get(
                (country, n), set()
            )
        )
    candidates["number"] = c

    # number + name4
    c = set()
    for n in numbers:
        c.update(
            indexes["number_name4"].get(
                (country, n, name4), set()
            )
        )
    candidates["number_name4"] = c

    # number + address5
    c = set()
    addr5 = first_chars(r.business_address_norm, 5)

    for n in numbers:
        c.update(
            indexes["number_addr5"].get(
                (country, n, addr5), set()
            )
        )
    candidates["number_addr5"] = c

    for key, cand in candidates.items():
        if true_ids & cand:
            recovered[key] += len(true_ids & cand)

print("\n========== BLOCK ANALYSIS ==========")
print("Total missed true pairs:", total_true)

for key, value in recovered.items():
    recall = value / total_true if total_true else 0

    print(
        f"{key:20s} "
        f"recovered={value:8d} "
        f"recall={recall:.4%}"
    )

print("====================================")
