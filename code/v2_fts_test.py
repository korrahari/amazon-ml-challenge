import sqlite3
import pandas as pd
from rapidfuzz.fuzz import ratio

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
retrieved_true = 0

for _, row in missed.iterrows():

    sid = row.source1_entity_id

    if sid not in s1.index:
        continue

    r = s1.loc[sid]

    name = str(r.business_name_norm)
    address = str(r.business_address_norm)

    # FTS5 search using name + address.
    # OR keeps retrieval recall high; later similarity filtering
    # will control precision.
    terms = []

    for token in name.split():
        token = token.replace('"', ' ')
        if len(token) >= 3:
            terms.append(token)

    query = " OR ".join(terms[:8])

    if not query:
        continue

    cur = con.execute(
        """
        SELECT target_id, name, address
        FROM targets
        WHERE targets MATCH ?
        LIMIT 100
        """,
        (query,)
    )

    retrieved = cur.fetchall()
    retrieved_ids = {x[0] for x in retrieved}

    true_ids = set(
        x.strip()
        for x in str(row.matched_entity_ids).split(",")
        if x.strip()
    )

    total_true += len(true_ids)
    retrieved_true += len(true_ids & retrieved_ids)

con.close()

print("Test S1:", len(missed))
print("True matches:", total_true)
print("Retrieved true matches:", retrieved_true)

if total_true:
    print(
        "Retrieval recall:",
        round(100 * retrieved_true / total_true, 2),
        "%"
    )
