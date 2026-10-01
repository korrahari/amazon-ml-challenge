import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz
from collections import defaultdict

MISSED = "output/v2_missed_s1.tsv"
S1_FILE = "data/normalized/train_source1.tsv"
S2_FILE = "data/normalized/train_source2.tsv"
S3_FILE = "data/normalized/train_source3.tsv"

OUT = "output/v2_recovered_matches.tsv"

BATCH_SIZE = 5000
TARGET_CHUNK = 100000

print("Loading missed S1 IDs...")

missed = pd.read_csv(MISSED, sep="\t")

wanted = set(missed["source1_entity_id"])

print("Difficult S1:", len(wanted))

# Load only the difficult S1 records.
s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    usecols=[
        "entity_id",
        "business_name_norm",
        "business_name_ascii",
        "business_name_compact",
        "business_address_norm",
        "business_address_compact",
        "country_norm",
    ]
)

s1 = s1[s1["entity_id"].isin(wanted)].copy()

print("Loaded difficult S1:", len(s1))

# Country groups for cheap blocking.
s1_by_country = defaultdict(list)

for i, row in s1.iterrows():
    country = str(row["country_norm"])
    name = str(row["business_name_compact"])

    if len(name) < 3:
        continue

    s1_by_country[country].append(i)

print("Countries represented:", len(s1_by_country))

# We will process targets country-by-country in chunks.
# This avoids loading all ~10M targets into memory.

results = []

def clean(x):
    if pd.isna(x):
        return ""
    return str(x)

def add_candidates(target_df, source):
    target_df["entity_id"] = target_df["entity_id"].astype(str)

    for country, s1_indices in s1_by_country.items():

        t = target_df[target_df["country_norm"] == country]

        if len(t) == 0:
            continue

        choices = t["business_name_compact"].fillna("").astype(str).tolist()

        if not choices:
            continue

        for start in range(0, len(s1_indices), BATCH_SIZE):

            inds = s1_indices[start:start + BATCH_SIZE]

            for idx in inds:
                row = s1.loc[idx]

                name = clean(row["business_name_compact"])

                if len(name) < 3:
                    continue

                # Retrieve only the strongest name candidates.
                matches = process.extract(
                    name,
                    choices,
                    scorer=fuzz.WRatio,
                    limit=3,
                    score_cutoff=70
                )

                for _, score, pos in matches:
                    target = t.iloc[pos]

                    addr1 = clean(row["business_address_compact"])
                    addr2 = clean(target["business_address_compact"])

                    addr_score = (
                        fuzz.ratio(addr1, addr2)
                        if addr1 and addr2
                        else 0
                    )

                    # Conservative recovery rule.
                    if score >= 85 and addr_score >= 45:
                        results.append({
                            "source1_entity_id": row["entity_id"],
                            "candidate_entity_id":
                                source + "-" + str(target["entity_id"]),
                            "source": source,
                            "name_score": score,
                            "address_score": addr_score,
                        })

            print(
                f"{source}: processed {min(start + BATCH_SIZE, len(s1_indices))}"
                f"/{len(s1_indices)} for country {country!r}"
            )

# IMPORTANT:
# We process one target file at a time.
for source, path in [
    ("S2", S2_FILE),
    ("S3", S3_FILE),
]:
    print("\nProcessing", source)

    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=[
            "entity_id",
            "business_name_compact",
            "business_address_compact",
            "country_norm",
        ],
        chunksize=TARGET_CHUNK,
    ):
        add_candidates(chunk, source)

        # Release chunk before loading the next one.
        del chunk

print("\nRecovered candidates:", len(results))

if results:
    out = pd.DataFrame(results).drop_duplicates(
        subset=["source1_entity_id", "candidate_entity_id"]
    )

    out.to_csv(OUT, sep="\t", index=False)

    print("Written:", OUT)
    print("Unique recovered:", len(out))
else:
    print("No recovered candidates.")
