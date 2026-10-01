import duckdb

DB_PATH = "blocking.duckdb"

print("=" * 70)
print("CHECKING BLOCKING RECALL")
print("=" * 70)

con = duckdb.connect(DB_PATH)


con.execute("SET memory_limit='6GB'")
con.execute("SET threads=2")
con.execute("SET preserve_insertion_order=false")
con.execute("SET temp_directory='duckdb_tmp'")
con.execute("SET max_temp_directory_size='8GB'")


# ------------------------------------------------------------
# 1. Read ground truth
# ------------------------------------------------------------

print("\n[1/4] Reading ground truth...")

con.execute("""
    CREATE OR REPLACE TABLE ground_truth AS

    SELECT
        source1_entity_id,
        matched_entity_ids

    FROM read_csv(
        'data/train_ground_truth.tsv',
        delim='\\t',
        header=true,
        nullstr=''
    )
""")

gt_rows = con.execute("""
    SELECT COUNT(*)
    FROM ground_truth
""").fetchone()[0]

print("Ground-truth Source 1 rows:", gt_rows)


# ------------------------------------------------------------
# 2. Convert comma-separated matches into individual pairs
# ------------------------------------------------------------

print("\n[2/4] Expanding true matches...")

con.execute("""
    CREATE OR REPLACE TABLE true_pairs AS

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

true_pair_count = con.execute("""
    SELECT COUNT(*)
    FROM true_pairs
""").fetchone()[0]

print("Total true match pairs:", true_pair_count)


# ------------------------------------------------------------
# 3. Check which true matches survived blocking
# ------------------------------------------------------------

print("\n[3/4] Comparing ground truth with candidates...")

con.execute("""
    CREATE OR REPLACE TABLE recall_check AS

    SELECT
        tp.source1_entity_id,
        tp.candidate_entity_id,

        CASE
            WHEN c.source1_entity_id IS NOT NULL
            THEN 1
            ELSE 0
        END AS survived_blocking

    FROM true_pairs tp

    LEFT JOIN candidates c
      ON tp.source1_entity_id = c.source1_entity_id
     AND tp.candidate_entity_id = c.candidate_entity_id
""")

survived = con.execute("""
    SELECT SUM(survived_blocking)
    FROM recall_check
""").fetchone()[0] or 0

missed = true_pair_count - survived

recall = (
    survived / true_pair_count
    if true_pair_count > 0
    else 0
)


# ------------------------------------------------------------
# 4. Source 1 level recall
# ------------------------------------------------------------

print("\n[4/4] Checking entity-level recall...")

con.execute("""
    CREATE OR REPLACE TABLE entity_recall AS

    SELECT
        tp.source1_entity_id,

        COUNT(*) AS true_match_count,

        SUM(
            CASE
                WHEN c.source1_entity_id IS NOT NULL
                THEN 1
                ELSE 0
            END
        ) AS captured_match_count

    FROM true_pairs tp

    LEFT JOIN candidates c
      ON tp.source1_entity_id = c.source1_entity_id
     AND tp.candidate_entity_id = c.candidate_entity_id

    GROUP BY tp.source1_entity_id
""")

entities_with_all_matches = con.execute("""
    SELECT COUNT(*)
    FROM entity_recall
    WHERE captured_match_count = true_match_count
""").fetchone()[0]

entities_with_missing_matches = con.execute("""
    SELECT COUNT(*)
    FROM entity_recall
    WHERE captured_match_count < true_match_count
""").fetchone()[0]


print("\n" + "=" * 70)
print("BLOCKING RECALL RESULTS")
print("=" * 70)

print("Total true match pairs:", true_pair_count)
print("True pairs captured:", survived)
print("True pairs missed:", missed)

print(
    "Pair-level blocking recall:",
    round(recall * 100, 4),
    "%"
)

print()
print("Entities with ALL true matches captured:",
      entities_with_all_matches)

print("Entities with at least one missing match:",
      entities_with_missing_matches)

print("=" * 70)

con.close()
