import sqlite3
import pandas as pd
import time
from rapidfuzz import process, fuzz

S1_FILE = "data/normalized/train_source1.tsv"
MISSED_FILE = "output/v2_missed_s1.tsv"
DB = "output/v2_trigram.sqlite"
OUT_FILE = "output/v2_trigram_recovered.tsv"

# --------------------------------------------------
# Load difficult S1 records
# --------------------------------------------------

missed = pd.read_csv(MISSED_FILE, sep="\t", dtype=str)
missed_ids = set(missed["source1_entity_id"])

s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str,
    usecols=[
        "entity_id",
        "business_name_norm",
        "business_address_norm",
        "country_norm"
    ]
).fillna("")

s1 = s1[s1.entity_id.isin(missed_ids)].copy()

print("Difficult S1:", len(s1))

# --------------------------------------------------
# Load target records ONCE from SQLite
# --------------------------------------------------

print("Loading target index...")

con = sqlite3.connect(DB)

targets = pd.read_sql_query(
    """
    SELECT entity_id, country, text
    FROM targets
    """,
    con
)

con.close()

targets = targets.fillna("")

print("Targets loaded:", len(targets))

# Make country buckets
country_targets = {}

for country, group in targets.groupby("country", sort=False):
    country_targets[country] = group.reset_index(drop=True)

print("Countries indexed:", len(country_targets))

# --------------------------------------------------
# Recovery
# --------------------------------------------------

results = []
start = time.time()

for i, r in enumerate(s1.itertuples(index=False), 1):

    sid = r.entity_id
    name = r.business_name_norm
    address = r.business_address_norm
    country = r.country_norm

    group = country_targets.get(country)

    if group is None or len(group) == 0:
        continue

    # Search name against target combined text.
    query = (name + " " + address).strip()

    if not query:
        continue

    choices = group["text"].tolist()

    matches = process.extract(
        query,
        choices,
        scorer=fuzz.WRatio,
        limit=10,
        score_cutoff=70
    )

    for _, score, idx in matches:

        if score < 75:
            continue

        tid = group.iloc[idx]["entity_id"]

        results.append(
            (
                sid,
                tid,
                round(score, 2)
            )
        )

    if i % 500 == 0:
        elapsed = time.time() - start

        print(
            f"{i}/{len(s1)} "
            f"({i/len(s1)*100:.1f}%) "
            f"recovered={len(results)} "
            f"time={elapsed/60:.1f} min"
        )

# --------------------------------------------------
# Save
# --------------------------------------------------

result = pd.DataFrame(
    results,
    columns=[
        "source1_entity_id",
        "matched_entity_id",
        "score"
    ]
)

result.to_csv(
    OUT_FILE,
    sep="\t",
    index=False
)

print()
print("RECOVERY COMPLETE")
print("Recovered rows:", len(result))
print("Output:", OUT_FILE)
print(
    "Time:",
    round((time.time() - start) / 60, 2),
    "minutes"
)

