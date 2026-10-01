import sqlite3
import time
from collections import defaultdict

DB = "output/v3_index.sqlite"

conn = sqlite3.connect(DB)
conn.execute("PRAGMA cache_size=-200000")
conn.execute("PRAGMA temp_store=FILE")

print("Loading rare relevant keys...")

rare_keys = set(
    conn.execute("""
        SELECT f.country, f.token5
        FROM token_frequency f
        JOIN (
            SELECT DISTINCT country, token5
            FROM difficult_s1_keys
        ) d
        ON f.country = d.country
        AND f.token5 = d.token5
        WHERE f.frequency <= 25
    """).fetchall()
)

print("Rare keys:", len(rare_keys))

# Map rare key -> difficult S1 IDs
print("Building key -> S1 index...")

key_to_s1 = defaultdict(set)

for country, token5, source1_entity_id in conn.execute("""
    SELECT country, token5, source1_entity_id
    FROM difficult_s1_keys
"""):
    if (country, token5) in rare_keys:
        key_to_s1[(country, token5)].add(source1_entity_id)

print("Index keys:", len(key_to_s1))

print("Scanning target index...")
start = time.time()

target_count = 0
candidate_pairs = 0
covered_s1 = set()

cur = conn.execute("""
    SELECT country, token5, entity_id
    FROM address_index
""")

for country, token5, target_id in cur:
    s1s = key_to_s1.get((country, token5))
    if s1s:
        target_count += 1
        candidate_pairs += len(s1s)
        covered_s1.update(s1s)

print()
print("=" * 60)
print("V3 RARE CANDIDATE COUNT")
print("=" * 60)
print("Target records matching rare keys:", target_count)
print("Candidate pairs:", f"{candidate_pairs:,}")
print("Difficult S1 covered:", f"{len(covered_s1):,}")
print("Average candidates per covered S1:",
      round(candidate_pairs / len(covered_s1), 2) if covered_s1 else 0)
print("Time:", round(time.time() - start, 1), "sec")
print("=" * 60)

conn.close()
