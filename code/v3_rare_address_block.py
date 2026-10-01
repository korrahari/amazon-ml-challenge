import duckdb

DB = "v3_rare_address_block.duckdb"

con = duckdb.connect(DB)

con.execute("SET memory_limit='5GB'")
con.execute("SET threads=2")
con.execute("SET preserve_insertion_order=false")
con.execute("SET temp_directory='duckdb_tmp_v3_rare'")
con.execute("SET max_temp_directory_size='4GB'")

print("=" * 70)
print("V3 — RARE ADDRESS TOKEN BLOCK RECALL")
print("=" * 70)

# --------------------------------------------------
# 1. Ground truth
# --------------------------------------------------

print("\n[1] Loading ground truth...")

con.execute("""
CREATE TEMP TABLE truth AS
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

print("True pairs:", f"{true_count:,}")

# --------------------------------------------------
# 2. Target token frequencies
# --------------------------------------------------

print("\n[2] Building target token frequencies...")

con.execute("""
CREATE TEMP TABLE target_token_freq AS

SELECT
    country_norm,
    LEFT(tok, 5) AS token5,
    COUNT(DISTINCT entity_id) AS freq
FROM (
    SELECT
        entity_id,
        country_norm,
        business_address_norm
    FROM read_csv(
        'data/normalized/train_source2.tsv',
        delim='\t',
        header=true,
        nullstr=''
    )

    UNION ALL

    SELECT
        entity_id,
        country_norm,
        business_address_norm
    FROM read_csv(
        'data/normalized/train_source3.tsv',
        delim='\t',
        header=true,
        nullstr=''
    )
) t,
UNNEST(
    regexp_extract_all(
        lower(t.business_address_norm),
        '[a-z0-9]+'
    )
) x(tok)
WHERE length(tok) >= 5
GROUP BY country_norm, LEFT(tok, 5)
""")

print(
    "Unique country/token keys:",
    f"{con.execute('SELECT COUNT(*) FROM target_token_freq').fetchone()[0]:,}"
)

# --------------------------------------------------
# 3. Test several rarity limits
# --------------------------------------------------

for limit in [50, 100, 250, 500, 1000]:

    print()
    print("-" * 70)
    print("Frequency limit:", limit)

    # --------------------------------------------------
    # S1 tokens
    # --------------------------------------------------

    con.execute("DROP TABLE IF EXISTS s1_tokens")

    con.execute(f"""
    CREATE TEMP TABLE s1_tokens AS
    SELECT DISTINCT
        s.entity_id AS source1_entity_id,
        s.country_norm,
        LEFT(tok, 5) AS token5
    FROM read_csv(
        'data/normalized/train_source1.tsv',
        delim='\\t',
        header=true,
        nullstr=''
    ) s,
    UNNEST(
        regexp_extract_all(
            lower(s.business_address_norm),
            '[a-z0-9]+'
        )
    ) x(tok)

    JOIN target_token_freq f
      ON s.country_norm = f.country_norm
     AND LEFT(tok, 5) = f.token5

    WHERE length(tok) >= 5
      AND f.freq <= {limit}
    """)

    # --------------------------------------------------
    # Recall only: join against true target IDs
    # --------------------------------------------------

    captured = con.execute("""
    SELECT COUNT(*)
    FROM (
        SELECT DISTINCT
            tr.source1_entity_id,
            tr.target_id
        FROM truth tr

        JOIN s1_tokens s
          ON tr.source1_entity_id = s.source1_entity_id

        JOIN (
            SELECT DISTINCT
                entity_id AS target_id,
                country_norm,
                LEFT(tok, 5) AS token5
            FROM (
                SELECT
                    entity_id,
                    country_norm,
                    business_address_norm
                FROM read_csv(
                    'data/normalized/train_source2.tsv',
                    delim='\t',
                    header=true,
                    nullstr=''
                )

                UNION ALL

                SELECT
                    entity_id,
                    country_norm,
                    business_address_norm
                FROM read_csv(
                    'data/normalized/train_source3.tsv',
                    delim='\t',
                    header=true,
                    nullstr=''
                )
            ) t,
            UNNEST(
                regexp_extract_all(
                    lower(t.business_address_norm),
                    '[a-z0-9]+'
                )
            ) x(tok)
            WHERE length(tok) >= 5
        ) target
          ON tr.target_id = target.target_id
         AND s.country_norm = target.country_norm
         AND s.token5 = target.token5
    )
    """).fetchone()[0]

    recall = captured / true_count * 100

    print("Captured true pairs:", f"{captured:,}")
    print("Recall:", f"{recall:.4f}%")

print()
print("=" * 70)
print("DONE")
print("=" * 70)

con.close()
