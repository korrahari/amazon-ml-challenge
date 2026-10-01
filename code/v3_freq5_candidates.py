import sqlite3
import time

DB = "output/v3_index.sqlite"
OUT = "output/v3_freq5_candidates.tsv"

conn = sqlite3.connect(DB)
conn.execute("PRAGMA cache_size=-200000")

keys = set(
    conn.execute("""
        SELECT country, token5
        FROM token_frequency
        WHERE frequency <= 5
    """).fetchall()
)

key_to_s1 = {}

for country, token5, s1_id in conn.execute("""
    SELECT country, token5, source1_entity_id
    FROM difficult_s1_keys
"""):
    key = (country, token5)
    if key in keys:
        key_to_s1.setdefault(key, set()).add(s1_id)

print("Relevant keys:", len(key_to_s1))

count = 0
start = time.time()

with open(OUT, "w") as f:
    f.write("source1_entity_id\tcandidate_entity_id\tsource\n")

    for country, token5, target_id in conn.execute("""
        SELECT country, token5, entity_id
        FROM address_index
    """):
        s1s = key_to_s1.get((country, token5))

        if not s1s:
            continue

        source = "S2" if target_id.startswith("S2-") else "S3"

        for s1_id in s1s:
            f.write(
                f"{s1_id}\t{target_id}\t{source}\n"
            )
            count += 1

print()
print("=" * 60)
print("V3 FREQ<=5 CANDIDATES")
print("=" * 60)
print("Candidates:", f"{count:,}")
print("Output:", OUT)
print("Time:", round(time.time() - start, 1), "sec")
print("=" * 60)

conn.close()
