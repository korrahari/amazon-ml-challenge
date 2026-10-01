import sqlite3
import pandas as pd
from rapidfuzz.fuzz import ratio, WRatio

DB = "output/v2_fts.sqlite"

missed = pd.read_csv(
    "output/v2_missed_s1.tsv",
    sep="\t"
).head(100)

s1 = pd.read_csv(
    "data/normalized/train_source1.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "country_norm",
        "business_name_norm",
        "business_address_norm"
    ]
)

s1 = s1[s1.entity_id.isin(set(missed.source1_entity_id))]
s1 = s1.set_index("entity_id")

con = sqlite3.connect(DB)

total_true = 0
retrieved_100 = 0
retrieved_500 = 0
retrieved_1000 = 0

for i, row in missed.iterrows():

    sid = row.source1_entity_id

    if sid not in s1.index:
        continue

    r = s1.loc[sid]

    name = str(r.business_name_norm)
    address = str(r.business_address_norm)
    country = str(r.country_norm)

    tokens = [
        t.replace('"', ' ')
        for t in name.split()
        if len(t) >= 3
    ]

    if not tokens:
        continue

    query = " OR ".join(tokens[:8])

    cur = con.execute(
        """
        SELECT target_id, name, address
        FROM targets
        WHERE country = ?
          AND targets MATCH ?
        LIMIT 1000
        """,
        (country, query)
    )

    retrieved = cur.fetchall()

    # Fuzzy-rank the FTS candidates
    scored = []

    for target_id, target_name, target_address in retrieved:

        name_score = WRatio(name, str(target_name))
        address_score = ratio(address, str(target_address))

        combined = 0.65 * name_score + 0.35 * address_score

        scored.append(
            (combined, target_id)
        )

    scored.sort(reverse=True)

    ids100 = {x[1] for x in scored[:100]}
    ids500 = {x[1] for x in scored[:500]}
    ids1000 = {x[1] for x in scored[:1000]}

    true_ids = {
        x.strip()
        for x in str(row.matched_entity_ids).split(",")
        if x.strip()
    }

    total_true += len(true_ids)

    retrieved_100 += len(true_ids & ids100)
    retrieved_500 += len(true_ids & ids500)
    retrieved_1000 += len(true_ids & ids1000)

con.close()

print()
print("Test S1:", len(missed))
print("True matches:", total_true)
print("Retrieved @100:", retrieved_100)
print("Retrieved @500:", retrieved_500)
print("Retrieved @1000:", retrieved_1000)

if total_true:
    print()
    print("Recall @100 :", round(100 * retrieved_100 / total_true, 2), "%")
    print("Recall @500 :", round(100 * retrieved_500 / total_true, 2), "%")
    print("Recall @1000:", round(100 * retrieved_1000 / total_true, 2), "%")
