import duckdb

DB = "v3_address_candidate_count.duckdb"

con = duckdb.connect(DB)

con.execute("SET memory_limit='6GB'")
con.execute("SET threads=2")
con.execute("SET preserve_insertion_order=false")
con.execute("SET temp_directory='duckdb_tmp_v3_count'")
con.execute("SET max_temp_directory_size='6GB'")

print("=" * 70)
print("V3 — ADDRESS TOKEN CANDIDATE COUNT")
print("=" * 70)

print("\n[1] Building S1 address-token keys...")

con.execute("""
CREATE TEMP TABLE s1_tokens AS
SELECT DISTINCT
    s.entity_id AS source1_entity_id,
    s.country_norm,
    LEFT(tok, 5) AS token5
FROM read_csv(
    'data/normalized/train_source1.tsv',
    delim='\t',
    header=true,
    nullstr=''
) s,
UNNEST(
    regexp_extract_all(
        lower(s.business_address_norm),
        '[a-z0-9]+'
    )
) x(tok)
WHERE length(tok) >= 5
""")

print("\n[2] Building target address-token keys...")

con.execute("""
CREATE TEMP TABLE target_tokens AS

SELECT DISTINCT
    t.entity_id AS target_id,
    t.country_norm,
    LEFT(tok, 5) AS token5
FROM read_csv(
    'data/normalized/train_source2.tsv',
    delim='\t',
    header=true,
    nullstr=''
) t,
UNNEST(
    regexp_extract_all(
        lower(t.business_address_norm),
        '[a-z0-9]+'
    )
) x(tok)
WHERE length(tok) >= 5

UNION

SELECT DISTINCT
    t.entity_id AS target_id,
    t.country_norm,
    LEFT(tok, 5) AS token5
FROM read_csv(
    'data/normalized/train_source3.tsv',
    delim='\t',
    header=true,
    nullstr=''
) t,
UNNEST(
    regexp_extract_all(
        lower(t.business_address_norm),
        '[a-z0-9]+'
    )
) x(tok)
WHERE length(tok) >= 5
""")

print("\n[3] Counting candidate pairs...")

result = con.execute("""
SELECT COUNT(*)
FROM (
    SELECT DISTINCT
        s.source1_entity_id,
        t.target_id
    FROM s1_tokens s
    JOIN target_tokens t
      ON s.country_norm = t.country_norm
     AND s.token5 = t.token5
)
""").fetchone()[0]

print()
print("=" * 70)
print("RESULT")
print("=" * 70)
print("Address-token candidate pairs:", f"{result:,}")

s1_count = con.execute("""
SELECT COUNT(DISTINCT source1_entity_id)
FROM s1_tokens
""").fetchone()[0]

print("S1 entities covered:", f"{s1_count:,}")

print("Average candidates per covered S1:",
      round(result / s1_count, 2))

print("=" * 70)

con.close()
