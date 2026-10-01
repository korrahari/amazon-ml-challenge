import duckdb
from pathlib import Path


BASE = Path(".")

DB_PATH = BASE / "blocking.duckdb"
OUTPUT_DIR = BASE / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

NORMALIZED_DIR = BASE / "data" / "normalized"

S1 = NORMALIZED_DIR / "train_source1.tsv"
S2 = NORMALIZED_DIR / "train_source2.tsv"
S3 = NORMALIZED_DIR / "train_source3.tsv"


con = duckdb.connect(str(DB_PATH))

con.execute("PRAGMA memory_limit='7GB'")
con.execute("PRAGMA threads=2")
con.execute("PRAGMA preserve_insertion_order=false")
con.execute("PRAGMA temp_directory='duckdb_tmp'")
con.execute("PRAGMA max_temp_directory_size='6GB'")


print("=" * 70)
print("BUILDING MULTI-BLOCK CANDIDATES")
print("=" * 70)


# ------------------------------------------------------------
# READ SOURCE 1
# ------------------------------------------------------------

print("\nLoading Source 1...")

con.execute(f"""
CREATE OR REPLACE TABLE s1 AS
SELECT
    entity_id AS source1_entity_id,
    business_name_norm,
    business_name_compact,
    business_name_token_sorted,
    business_address_norm,
    business_address_compact,
    business_address_token_sorted,
    house_number,
    country_norm
FROM read_csv(
    '{S1}',
    delim='\\t',
    header=true,
    all_varchar=true
)
""")


print("Source 1 loaded.")


# ------------------------------------------------------------
# READ SOURCE 2 + SOURCE 3
# ------------------------------------------------------------

print("\nLoading Source 2...")

con.execute(f"""
CREATE OR REPLACE TABLE s2 AS
SELECT
    entity_id AS candidate_entity_id,
    business_name_norm,
    business_name_compact,
    business_name_token_sorted,
    business_address_norm,
    business_address_compact,
    business_address_token_sorted,
    house_number,
    country_norm,
    'S2' AS source
FROM read_csv(
    '{S2}',
    delim='\\t',
    header=true,
    all_varchar=true
)
""")


print("Source 2 loaded.")


print("\nLoading Source 3...")

con.execute(f"""
CREATE OR REPLACE TABLE s3 AS
SELECT
    entity_id AS candidate_entity_id,
    business_name_norm,
    business_name_compact,
    business_name_token_sorted,
    business_address_norm,
    business_address_compact,
    business_address_token_sorted,
    house_number,
    country_norm,
    'S3' AS source
FROM read_csv(
    '{S3}',
    delim='\\t',
    header=true,
    all_varchar=true
)
""")


print("Source 3 loaded.")


# ------------------------------------------------------------
# COMBINE TARGET SOURCES
# ------------------------------------------------------------

print("\nCombining Source 2 + Source 3...")

con.execute("""
CREATE OR REPLACE TABLE targets AS
SELECT * FROM s2
UNION ALL
SELECT * FROM s3
""")


# ------------------------------------------------------------
# BLOCK 1
# COUNTRY + EXACT COMPACT NAME
# ------------------------------------------------------------

print("\nCreating Block 1: country + exact compact name...")

con.execute("""
CREATE OR REPLACE TABLE block1 AS
SELECT
    s.source1_entity_id,
    t.candidate_entity_id,
    t.source
FROM s1 s
JOIN targets t
    ON s.country_norm = t.country_norm
   AND s.business_name_compact <> ''
   AND s.business_name_compact = t.business_name_compact
""")


print("Block 1 complete.")


# ------------------------------------------------------------
# BLOCK 2
# COUNTRY + TOKEN-SORTED NAME
# ------------------------------------------------------------

print("\nCreating Block 2: country + token-sorted name...")

con.execute("""
CREATE OR REPLACE TABLE block2 AS
WITH sizes AS (
    SELECT
        country_norm,
        business_name_token_sorted,
        COUNT(*) AS block_size
    FROM targets
    WHERE business_name_token_sorted <> ''
    GROUP BY country_norm, business_name_token_sorted
)
SELECT
    s.source1_entity_id,
    t.candidate_entity_id,
    t.source
FROM s1 s
JOIN targets t
    ON s.country_norm = t.country_norm
   AND s.business_name_token_sorted <> ''
   AND s.business_name_token_sorted = t.business_name_token_sorted
JOIN sizes z
    ON t.country_norm = z.country_norm
   AND t.business_name_token_sorted = z.business_name_token_sorted
WHERE z.block_size <= 100
""")


print("Block 2 complete.")


# ------------------------------------------------------------
# BLOCK 3
# COUNTRY + HOUSE NUMBER + NAME PREFIX
# ------------------------------------------------------------

print("\nCreating Block 3: country + house number + name prefix...")

con.execute("""
CREATE OR REPLACE TABLE block3 AS
WITH sizes AS (
    SELECT
        country_norm,
        house_number,
        LEFT(business_name_compact, 3) AS name_prefix,
        COUNT(*) AS block_size
    FROM targets
    WHERE house_number <> ''
      AND business_name_compact <> ''
    GROUP BY
        country_norm,
        house_number,
        LEFT(business_name_compact, 3)
)
SELECT
    s.source1_entity_id,
    t.candidate_entity_id,
    t.source
FROM s1 s
JOIN targets t
    ON s.country_norm = t.country_norm
   AND s.house_number <> ''
   AND s.house_number = t.house_number
   AND LEFT(s.business_name_compact, 3)
       = LEFT(t.business_name_compact, 3)
JOIN sizes z
    ON t.country_norm = z.country_norm
   AND t.house_number = z.house_number
   AND LEFT(t.business_name_compact, 3) = z.name_prefix
WHERE z.block_size <= 100
""")


print("Block 3 complete.")


# ------------------------------------------------------------
# BLOCK 4
# COUNTRY + HOUSE NUMBER + ADDRESS PREFIX
# ------------------------------------------------------------

print("\nCreating Block 4: country + house number + address prefix...")

con.execute("""
CREATE OR REPLACE TABLE block4 AS
WITH sizes AS (
    SELECT
        country_norm,
        house_number,
        LEFT(business_address_compact, 6) AS address_prefix,
        COUNT(*) AS block_size
    FROM targets
    WHERE house_number <> ''
      AND business_address_compact <> ''
    GROUP BY
        country_norm,
        house_number,
        LEFT(business_address_compact, 6)
)
SELECT
    s.source1_entity_id,
    t.candidate_entity_id,
    t.source
FROM s1 s
JOIN targets t
    ON s.country_norm = t.country_norm
   AND s.house_number <> ''
   AND s.house_number = t.house_number
   AND LEFT(s.business_address_compact, 6)
       = LEFT(t.business_address_compact, 6)
JOIN sizes z
    ON t.country_norm = z.country_norm
   AND t.house_number = z.house_number
   AND LEFT(t.business_address_compact, 6) = z.address_prefix
WHERE z.block_size <= 100
""")


print("Block 4 complete.")


# ------------------------------------------------------------
# BLOCK 5
# COUNTRY + ADDRESS TOKEN SORT
# ------------------------------------------------------------

print("\nCreating Block 5: country + address token-sorted...")

con.execute("""
CREATE OR REPLACE TABLE block5 AS
WITH sizes AS (
    SELECT
        country_norm,
        business_address_token_sorted,
        COUNT(*) AS block_size
    FROM targets
    WHERE business_address_token_sorted <> ''
    GROUP BY country_norm, business_address_token_sorted
)
SELECT
    s.source1_entity_id,
    t.candidate_entity_id,
    t.source
FROM s1 s
JOIN targets t
    ON s.country_norm = t.country_norm
   AND s.business_address_token_sorted <> ''
   AND s.business_address_token_sorted
       = t.business_address_token_sorted
JOIN sizes z
    ON t.country_norm = z.country_norm
   AND t.business_address_token_sorted = z.business_address_token_sorted
WHERE z.block_size <= 50
""")


print("Block 5 complete.")

# ------------------------------------------------------------
# BLOCK 6
# COUNTRY + FIRST 4 CHARACTERS OF NAME
# ------------------------------------------------------------

print("\nCreating Block 6: country + first 4 name characters...")

con.execute("""
CREATE OR REPLACE TABLE block6 AS
WITH sizes AS (
    SELECT
        country_norm,
        LEFT(business_name_compact, 4) AS name_prefix,
        COUNT(*) AS block_size
    FROM targets
    WHERE business_name_compact <> ''
      AND LENGTH(business_name_compact) >= 4
    GROUP BY
        country_norm,
        LEFT(business_name_compact, 4)
)
SELECT
    s.source1_entity_id,
    t.candidate_entity_id,
    t.source
FROM s1 s
JOIN targets t
    ON s.country_norm = t.country_norm
   AND LENGTH(s.business_name_compact) >= 4
   AND LENGTH(t.business_name_compact) >= 4
   AND LEFT(s.business_name_compact, 4)
       = LEFT(t.business_name_compact, 4)
JOIN sizes z
    ON t.country_norm = z.country_norm
   AND LEFT(t.business_name_compact, 4) = z.name_prefix
WHERE z.block_size <= 200
""")


print("Block 6 complete.")


con.execute("""
CREATE OR REPLACE TABLE candidates AS
SELECT source1_entity_id, candidate_entity_id, source FROM block1

UNION

SELECT source1_entity_id, candidate_entity_id, source FROM block2

UNION

SELECT source1_entity_id, candidate_entity_id, source FROM block3

UNION

SELECT source1_entity_id, candidate_entity_id, source FROM block4

UNION

SELECT source1_entity_id, candidate_entity_id, source FROM block5

UNION

SELECT source1_entity_id, candidate_entity_id, source FROM block6
""")

# ------------------------------------------------------------
# STATISTICS
# ------------------------------------------------------------

total_s1 = con.execute("""
SELECT COUNT(*) FROM s1
""").fetchone()[0]

covered_s1 = con.execute("""
SELECT COUNT(DISTINCT source1_entity_id)
FROM candidates
""").fetchone()[0]

total_candidates = con.execute("""
SELECT COUNT(*)
FROM candidates
""").fetchone()[0]

max_candidates = con.execute("""
SELECT MAX(cnt)
FROM (
    SELECT source1_entity_id, COUNT(*) AS cnt
    FROM candidates
    GROUP BY source1_entity_id
)
""").fetchone()[0]

avg_candidates = con.execute("""
SELECT AVG(cnt)
FROM (
    SELECT source1_entity_id, COUNT(*) AS cnt
    FROM candidates
    GROUP BY source1_entity_id
)
""").fetchone()[0]


print("\n" + "=" * 70)
print("BLOCKING STATISTICS")
print("=" * 70)

print(f"Total Source 1:              {total_s1:,}")
print(f"Source 1 with candidates:    {covered_s1:,}")
print(f"Total candidate pairs:       {total_candidates:,}")
print(f"Average candidates/covered:  {avg_candidates:.2f}")
print(f"Maximum candidates:           {max_candidates:,}")


# ------------------------------------------------------------
# EXPORT RAW CANDIDATES
# ------------------------------------------------------------

output_file = OUTPUT_DIR / "train_candidate_pairs_raw.tsv"

print("\nExporting candidate pairs...")

con.execute(f"""
COPY (
    SELECT
        source1_entity_id,
        candidate_entity_id,
        source
    FROM candidates
    ORDER BY source1_entity_id
)
TO '{output_file}'
WITH (
    FORMAT CSV,
    DELIMITER '\\t',
    HEADER TRUE
)
""")


print(f"\nSaved: {output_file}")


# ------------------------------------------------------------
# CLOSE
# ------------------------------------------------------------

con.close()

print("\n" + "=" * 70)
print("BLOCKING COMPLETE")
print("=" * 70)
