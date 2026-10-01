import sqlite3
import pandas as pd
import os
import time

DB = "output/v2_trigram.sqlite"

if os.path.exists(DB):
    os.remove(DB)

con = sqlite3.connect(DB)

con.execute("""
CREATE VIRTUAL TABLE targets USING fts5(
    entity_id UNINDEXED,
    country UNINDEXED,
    text,
    tokenize='trigram'
)
""")

con.commit()

for source, path in [
    ("S2", "data/normalized/train_source2.tsv"),
    ("S3", "data/normalized/train_source3.tsv")
]:
    print("Indexing", source)
    start = time.time()

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=100_000,
        usecols=[
            "entity_id",
            "country_norm",
            "business_name_norm",
            "business_address_norm"
        ]
    ):
        chunk = chunk.fillna("")

        rows = []
        for r in chunk.itertuples(index=False):
            text = (
                str(r.business_name_norm) + " " +
                str(r.business_address_norm)
            ).lower()

            rows.append((
                str(r.entity_id),
                str(r.country_norm),
                text
            ))

        con.executemany(
            "INSERT INTO targets(entity_id,country,text) VALUES (?,?,?)",
            rows
        )
        con.commit()

    print(
        source,
        "done in",
        round(time.time() - start, 1),
        "seconds"
    )

con.close()

print("TRIGRAM INDEX COMPLETE")
