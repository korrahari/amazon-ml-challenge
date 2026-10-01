import duckdb

DB = "v3_address_block.duckdb"

con = duckdb.connect(DB)

con.execute("SET memory_limit='6GB'")
con.execute("SET threads=2")
con.execute("SET preserve_insertion_order=false")
con.execute("SET temp_directory='duckdb_tmp_v3_addr'")
con.execute("SET max_temp_directory_size='6GB'")

print("=" * 70)
print("V3 — ADDRESS TOKEN BLOCK RECALL")
print("=" * 70)

# --------------------------------------------------
# Ground truth
# --------------------------------------------------

print("\n[1] Loading ground truth...")

con.execute("""
CREATE OR REPLACE TEMP TABLE truth AS
SELECT
    source1_entity_id,
    trim(match_id) AS target_id
FROM read_csv(
    'data/train_ground_truth.tsv',
    delim='\t',
    header=true,
    nullstr=''
),
UNNEST(
    CASE
        WHEN matched_entity_ids = ''
        THEN []
        ELSE string_split(matched_entity_ids, ',')
    END
) AS t(match_id)
WHERE trim(match_id) <> ''
""")

true_count = con.execute(
    "SELECT COUNT(*) FROM truth"
).fetchone()[0]

print("True pairs:", true_count)

# --------------------------------------------------
# S1 address tokens
# --------------------------------------------------

print("\n[2] Building S1 address-token keys...")

con.execute("""
CREATE OR REPLACE TEMP TABLE s1_tokens AS
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
) t(tok)
WHERE length(tok) >= 5
""")

print(
    "S1 token keys:",
    con.execute("SELECT COUNT(*) FROM s1_tokens").fetchone()[0]
)

# --------------------------------------------------
# Target address tokens
# --------------------------------------------------

print("\n[3] Building target address-token keys...")

con.execute("""
CREATE OR REPLACE TEMP TABLE target_tokens AS

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

print(
    "Target token keys:",
    con.execute("SELECT COUNT(*) FROM target_tokens").fetchone()[0]
)

# --------------------------------------------------
# Test recall WITHOUT materializing all candidates
# --------------------------------------------------

print("\n[4] Measuring true-pair recall...")

con.execute("""
CREATE OR REPLACE TEMP TABLE truth_token AS
SELECT DISTINCT
    tr.source1_entity_id,
    tr.target_id
FROM truth tr

JOIN s1_tokens s
    ON tr.source1_entity_id = s.source1_entity_id

JOIN target_tokens t
    ON tr.target_id = t.target_id
   AND s.country_norm = t.country_norm
   AND s.token5 = t.token5
""")

captured = con.execute(
    "SELECT COUNT(*) FROM truth_token"
).fetchone()[0]

recall = (
    captured / true_count * 100
    if true_count else 0
)

print()
print("=" * 70)
print("RESULT")
print("=" * 70)
print("True pairs:", true_count)
print("Captured:", captured)
print("Pair recall:", round(recall, 4), "%")
print("=" * 70)

con.close()
