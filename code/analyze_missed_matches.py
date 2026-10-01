import duckdb

DB_PATH = "blocking.duckdb"

print("=" * 70)
print("ANALYZING MISSED TRUE MATCHES")
print("=" * 70)

con = duckdb.connect(DB_PATH)

con.execute("SET memory_limit='6GB'")
con.execute("SET threads=2")
con.execute("SET preserve_insertion_order=false")
con.execute("SET temp_directory='duckdb_tmp'")
con.execute("SET max_temp_directory_size='8GB'")

print("\n[1/5] Loading ground truth...")

con.execute("""
CREATE OR REPLACE TEMP TABLE ground_truth AS
SELECT
    source1_entity_id,
    matched_entity_ids
FROM read_csv(
    'data/train_ground_truth.tsv',
    delim='\t',
    header=true,
    nullstr=''
)
""")

print("\n[2/5] Expanding true matches...")

con.execute("""
CREATE OR REPLACE TEMP TABLE true_pairs AS
SELECT
    source1_entity_id,
    trim(match_id) AS candidate_entity_id
FROM ground_truth,
UNNEST(
    CASE
        WHEN matched_entity_ids = ''
        THEN []
        ELSE string_split(matched_entity_ids, ',')
    END
) AS t(match_id)
WHERE trim(match_id) <> ''
""")

print("\n[3/5] Finding missed pairs...")

con.execute("""
CREATE OR REPLACE TEMP TABLE missed AS
SELECT
    tp.source1_entity_id,
    tp.candidate_entity_id
FROM true_pairs tp
LEFT JOIN candidates c
    ON tp.source1_entity_id = c.source1_entity_id
   AND tp.candidate_entity_id = c.candidate_entity_id
WHERE c.source1_entity_id IS NULL
""")

missed_count = con.execute("""
SELECT COUNT(*) FROM missed
""").fetchone()[0]

print("Missed true pairs:", missed_count)

print("\n[4/5] Loading normalized source data...")

con.execute("""
CREATE OR REPLACE TEMP TABLE s1 AS
SELECT
    entity_id,
    business_name,
    business_address,
    country,
    business_name_ascii,
    business_address_ascii
FROM read_csv(
    'data/normalized/train_source1.tsv',
    delim='\t',
    header=true,
    nullstr=''
)
""")

con.execute("""
CREATE OR REPLACE TEMP TABLE s2 AS
SELECT
    entity_id,
    business_name,
    business_address,
    country,
    business_name_ascii,
    business_address_ascii
FROM read_csv(
    'data/normalized/train_source2.tsv',
    delim='\t',
    header=true,
    nullstr=''
)
""")

con.execute("""
CREATE OR REPLACE TEMP TABLE s3 AS
SELECT
    entity_id,
    business_name,
    business_address,
    country,
    business_name_ascii,
    business_address_ascii
FROM read_csv(
    'data/normalized/train_source3.tsv',
    delim='\t',
    header=true,
    nullstr=''
)
""")

print("\n[5/5] Checking missed-pair characteristics...")

con.execute("""
CREATE OR REPLACE TEMP TABLE missed_details AS

SELECT
    m.source1_entity_id,
    m.candidate_entity_id,

    s1.country AS s1_country,
    s1.business_name_ascii AS s1_name,
    s1.business_address_ascii AS s1_address,

    COALESCE(s2.country, s3.country) AS candidate_country,
    COALESCE(s2.business_name_ascii, s3.business_name_ascii)
        AS candidate_name,
    COALESCE(s2.business_address_ascii, s3.business_address_ascii)
        AS candidate_address

FROM missed m

LEFT JOIN s1
    ON m.source1_entity_id = s1.entity_id

LEFT JOIN s2
    ON m.candidate_entity_id = s2.entity_id

LEFT JOIN s3
    ON m.candidate_entity_id = s3.entity_id
""")

print("\n--- Country agreement ---")

rows = con.execute("""
SELECT
    CASE
        WHEN s1_country = candidate_country
        THEN 'same_country'
        ELSE 'different_or_missing'
    END AS category,
    COUNT(*) AS pairs
FROM missed_details
GROUP BY category
ORDER BY pairs DESC
""").fetchall()

for category, count in rows:
    print(f"{category}: {count}")

print("\n--- Exact normalized name agreement ---")

rows = con.execute("""
SELECT
    CASE
        WHEN s1_name <> ''
         AND s1_name = candidate_name
        THEN 'exact_name'
        ELSE 'not_exact'
    END AS category,
    COUNT(*) AS pairs
FROM missed_details
GROUP BY category
ORDER BY pairs DESC
""").fetchall()

for category, count in rows:
    print(f"{category}: {count}")

print("\n--- Name first-4 agreement ---")

rows = con.execute("""
SELECT
    CASE
        WHEN length(s1_name) >= 4
         AND length(candidate_name) >= 4
         AND substr(s1_name, 1, 4) = substr(candidate_name, 1, 4)
        THEN 'same_first_4'
        ELSE 'different'
    END AS category,
    COUNT(*) AS pairs
FROM missed_details
GROUP BY category
ORDER BY pairs DESC
""").fetchall()

for category, count in rows:
    print(f"{category}: {count}")

print("\n--- Address first-5 agreement ---")

rows = con.execute("""
SELECT
    CASE
        WHEN length(s1_address) >= 5
         AND length(candidate_address) >= 5
         AND substr(s1_address, 1, 5) = substr(candidate_address, 1, 5)
        THEN 'same_first_5'
        ELSE 'different'
    END AS category,
    COUNT(*) AS pairs
FROM missed_details
GROUP BY category
ORDER BY pairs DESC
""").fetchall()

for category, count in rows:
    print(f"{category}: {count}")

print("\n--- Country + exact name agreement ---")

rows = con.execute("""
SELECT
    CASE
        WHEN s1_country = candidate_country
         AND s1_name <> ''
         AND s1_name = candidate_name
        THEN 'same_country_and_name'
        ELSE 'other'
    END AS category,
    COUNT(*) AS pairs
FROM missed_details
GROUP BY category
ORDER BY pairs DESC
""").fetchall()

for category, count in rows:
    print(f"{category}: {count}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

con.close()
