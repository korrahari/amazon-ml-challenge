import duckdb
from pathlib import Path

DB = "v3_sql_fast.duckdb"
OUT = Path("output/v3_sql_recovered.tsv")

con = duckdb.connect(DB)

con.execute("SET memory_limit='5GB'")
con.execute("SET threads=2")
con.execute("SET preserve_insertion_order=false")
con.execute("SET temp_directory='duckdb_tmp_v3_sql'")
con.execute("SET max_temp_directory_size='4GB'")

print("=" * 70)
print("V3 — SQL FAST RECOVERY")
print("=" * 70)

# ------------------------------------------------------------
# 1. Difficult S1 records
# ------------------------------------------------------------

print("\n[1] Loading difficult S1 records...")

con.execute("""
CREATE OR REPLACE TEMP TABLE difficult_s1 AS
SELECT
    s.entity_id AS source1_entity_id,
    s.country_norm,
    s.business_name_norm,
    s.business_address_norm,
    regexp_extract_all(
        lower(s.business_address_norm),
        '[a-z0-9]+'
    ) AS address_tokens
FROM read_csv(
    'data/normalized/train_source1.tsv',
    delim='\t',
    header=true,
    nullstr=''
) s
JOIN read_csv(
    'output/v2_missed_s1.tsv',
    delim='\t',
    header=true,
    nullstr=''
) d
ON s.entity_id = d.source1_entity_id
""")

print(
    "Difficult S1:",
    con.execute("SELECT COUNT(*) FROM difficult_s1").fetchone()[0]
)

# ------------------------------------------------------------
# 2. Create compact S1 address-token keys
# ------------------------------------------------------------

print("\n[2] Creating S1 address-token keys...")

con.execute("""
CREATE OR REPLACE TEMP TABLE s1_keys AS
SELECT DISTINCT
    d.source1_entity_id,
    d.country_norm,
    LEFT(token, 5) AS token5
FROM difficult_s1 d,
UNNEST(d.address_tokens) x(token)
WHERE length(token) >= 5
""")

print(
    "S1 keys:",
    con.execute("SELECT COUNT(*) FROM s1_keys").fetchone()[0]
)

# ------------------------------------------------------------
# 3. Stream Source 2 + Source 3 into compact target key tables
# ------------------------------------------------------------

print("\n[3] Creating target address-token keys...")

con.execute("""
CREATE OR REPLACE TEMP TABLE target_keys AS

SELECT DISTINCT
    t.entity_id AS target_id,
    'source2' AS source,
    t.country_norm,
    LEFT(token, 5) AS token5
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
) x(token)
WHERE length(token) >= 5

UNION

SELECT DISTINCT
    t.entity_id AS target_id,
    'source3' AS source,
    t.country_norm,
    LEFT(token, 5) AS token5
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
) x(token)
WHERE length(token) >= 5
""")

print(
    "Target keys:",
    f"{con.execute('SELECT COUNT(*) FROM target_keys').fetchone()[0]:,}"
)

# ------------------------------------------------------------
# 4. Candidate generation
# ------------------------------------------------------------

print("\n[4] Generating candidates...")

con.execute("""
CREATE OR REPLACE TEMP TABLE candidates AS
SELECT
    s.source1_entity_id,
    t.target_id,
    t.source,
    COUNT(*) AS shared_tokens
FROM s1_keys s
JOIN target_keys t
  ON s.country_norm = t.country_norm
 AND s.token5 = t.token5
GROUP BY
    s.source1_entity_id,
    t.target_id,
    t.source
HAVING COUNT(*) >= 2
""")

candidate_count = con.execute(
    "SELECT COUNT(*) FROM candidates"
).fetchone()[0]

print(
    "Candidates with >=2 shared address-token prefixes:",
    f"{candidate_count:,}"
)

# ------------------------------------------------------------
# 5. Add exact-name signal
# ------------------------------------------------------------

print("\n[5] Adding exact-name signal...")

con.execute("""
CREATE OR REPLACE TEMP TABLE scored AS
SELECT
    c.source1_entity_id,
    c.target_id,
    c.source,
    c.shared_tokens,

    CASE
        WHEN regexp_replace(
            lower(s.business_name_norm),
            '[^a-z0-9]',
            '',
            'g'
        )
        =
        regexp_replace(
            lower(t.business_name_norm),
            '[^a-z0-9]',
            '',
            'g'
        )
        THEN 1
        ELSE 0
    END AS exact_name

FROM candidates c

JOIN difficult_s1 s
  ON c.source1_entity_id = s.source1_entity_id

JOIN (
    SELECT
        entity_id,
        business_name_norm
    FROM read_csv(
        'data/normalized/train_source2.tsv',
        delim='\t',
        header=true,
        nullstr=''
    )

    UNION ALL

    SELECT
        entity_id,
        business_name_norm
    FROM read_csv(
        'data/normalized/train_source3.tsv',
        delim='\t',
        header=true,
        nullstr=''
    )
) t
  ON c.target_id = t.entity_id
""")

# ------------------------------------------------------------
# 6. Keep strong candidates
# ------------------------------------------------------------

print("\n[6] Filtering strong candidates...")

con.execute("""
CREATE OR REPLACE TEMP TABLE final_candidates AS
SELECT *
FROM scored
WHERE
      exact_name = 1
   OR shared_tokens >= 3
""")

final_count = con.execute(
    "SELECT COUNT(*) FROM final_candidates"
).fetchone()[0]

print(
    "Strong candidates:",
    f"{final_count:,}"
)

# ------------------------------------------------------------
# 7. Write
# ------------------------------------------------------------

print("\n[7] Writing V3 candidates...")

OUT.parent.mkdir(exist_ok=True)

con.execute(f"""
COPY final_candidates
TO '{OUT}'
WITH (
    HEADER,
    DELIMITER '\\t'
)
""")

print()
print("=" * 70)
print("V3 SQL FAST COMPLETE")
print("=" * 70)
print("Initial candidates:", f"{candidate_count:,}")
print("Strong candidates:", f"{final_count:,}")
print("Output:", OUT)
print("=" * 70)

con.close()
