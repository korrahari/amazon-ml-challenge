import duckdb
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from pathlib import Path


BASE = Path(".")
DB_PATH = BASE / "blocking.duckdb"
OUTPUT = BASE / "output" / "train_features.tsv"

SAMPLE_POSITIVES = 300_000
SAMPLE_NEGATIVES = 900_000


def similarity_features(row):

    name1 = row["s1_name"]
    name2 = row["target_name"]

    addr1 = row["s1_address"]
    addr2 = row["target_address"]

    name1_ascii = row["s1_name_ascii"]
    name2_ascii = row["target_name_ascii"]

    addr1_ascii = row["s1_address_ascii"]
    addr2_ascii = row["target_address_ascii"]

    # -------------------------
    # NAME FEATURES
    # -------------------------

    name_ratio = fuzz.ratio(name1, name2) / 100.0

    name_partial = fuzz.partial_ratio(name1, name2) / 100.0

    name_token_sort = fuzz.token_sort_ratio(
        name1,
        name2
    ) / 100.0

    name_token_set = fuzz.token_set_ratio(
        name1,
        name2
    ) / 100.0

    name_jaro = JaroWinkler.normalized_similarity(
        name1,
        name2
    )

    name_ascii_ratio = fuzz.ratio(
        name1_ascii,
        name2_ascii
    ) / 100.0

    name_exact = int(name1 == name2)

    name_compact_exact = int(
        row["s1_name_compact"] ==
        row["target_name_compact"]
    )

    name_token_exact = int(
        row["s1_name_token_sorted"] ==
        row["target_name_token_sorted"]
    )

    # -------------------------
    # ADDRESS FEATURES
    # -------------------------

    address_ratio = fuzz.ratio(
        addr1,
        addr2
    ) / 100.0

    address_partial = fuzz.partial_ratio(
        addr1,
        addr2
    ) / 100.0

    address_token_sort = fuzz.token_sort_ratio(
        addr1,
        addr2
    ) / 100.0

    address_token_set = fuzz.token_set_ratio(
        addr1,
        addr2
    ) / 100.0

    address_ascii_ratio = fuzz.ratio(
        addr1_ascii,
        addr2_ascii
    ) / 100.0

    address_exact = int(addr1 == addr2)

    address_token_exact = int(
        row["s1_address_token_sorted"] ==
        row["target_address_token_sorted"]
    )

    # -------------------------
    # STRUCTURAL FEATURES
    # -------------------------

    country_match = int(
        row["s1_country"] ==
        row["target_country"]
    )

    house_match = int(
        row["s1_house"] != "" and
        row["s1_house"] == row["target_house"]
    )

    name_length_diff = abs(
        len(name1) - len(name2)
    )

    address_length_diff = abs(
        len(addr1) - len(addr2)
    )

    # -------------------------
    # COMBINED FEATURES
    # -------------------------

    combined_score = (
        0.60 * name_ratio +
        0.40 * address_ratio
    )

    strong_name_address = int(
        name_ratio >= 0.85 and
        address_ratio >= 0.70
    )

    return {
        "name_ratio": name_ratio,
        "name_partial": name_partial,
        "name_token_sort": name_token_sort,
        "name_token_set": name_token_set,
        "name_jaro": name_jaro,
        "name_ascii_ratio": name_ascii_ratio,
        "name_exact": name_exact,
        "name_compact_exact": name_compact_exact,
        "name_token_exact": name_token_exact,

        "address_ratio": address_ratio,
        "address_partial": address_partial,
        "address_token_sort": address_token_sort,
        "address_token_set": address_token_set,
        "address_ascii_ratio": address_ascii_ratio,
        "address_exact": address_exact,
        "address_token_exact": address_token_exact,

        "country_match": country_match,
        "house_match": house_match,

        "name_length_diff": name_length_diff,
        "address_length_diff": address_length_diff,

        "combined_score": combined_score,
        "strong_name_address": strong_name_address,

        "source_is_s3": int(row["source"] == "S3"),

        "label": int(row["label"])
    }


def main():

    print("=" * 70)
    print("CREATING ML TRAINING FEATURES")
    print("=" * 70)

    con = duckdb.connect(str(DB_PATH))

    con.execute("PRAGMA memory_limit='7GB'")
    con.execute("PRAGMA threads=2")
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute("PRAGMA temp_directory='duckdb_tmp'")
    con.execute("PRAGMA max_temp_directory_size='6GB'")

    # ---------------------------------------------------------
    # Ground truth pairs
    # ---------------------------------------------------------

    print("\n[1/6] Loading ground truth...")

    con.execute("""
    CREATE OR REPLACE TEMP TABLE gt AS
    SELECT
        source1_entity_id,
        TRIM(x.target_id) AS target_id
    FROM read_csv(
        'data/train_ground_truth.tsv',
        delim='\t',
        header=true,
        all_varchar=true
    ),
    LATERAL UNNEST(
        STRING_SPLIT(
            COALESCE(matched_entity_ids, ''),
            ','
        )
    ) AS x(target_id)
    WHERE TRIM(x.target_id) <> ''
    """)

    gt_count = con.execute("""
        SELECT COUNT(*) FROM gt
    """).fetchone()[0]

    print(f"True pairs: {gt_count:,}")

    # ---------------------------------------------------------
    # Sample positives
    # ---------------------------------------------------------

    print("\n[2/6] Sampling positive pairs...")

    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE positive_pairs AS
    SELECT
        c.source1_entity_id,
        c.candidate_entity_id,
        c.source,
        1 AS label
    FROM candidates c
    INNER JOIN gt g
        ON c.source1_entity_id = g.source1_entity_id
       AND c.candidate_entity_id = g.target_id
    USING SAMPLE {SAMPLE_POSITIVES} ROWS (reservoir, 42)
    """)

    positive_count = con.execute("""
        SELECT COUNT(*) FROM positive_pairs
    """).fetchone()[0]

    print(f"Positive examples: {positive_count:,}")

    # ---------------------------------------------------------
    # Sample negatives
    # ---------------------------------------------------------

    print("\n[3/6] Sampling negative pairs...")

    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE negative_pairs AS
    SELECT
        c.source1_entity_id,
        c.candidate_entity_id,
        c.source,
        0 AS label
    FROM candidates c
    WHERE NOT EXISTS (
        SELECT 1
        FROM gt g
        WHERE g.source1_entity_id = c.source1_entity_id
          AND g.target_id = c.candidate_entity_id
    )
    USING SAMPLE {SAMPLE_NEGATIVES} ROWS (reservoir, 43)
    """)

    negative_count = con.execute("""
        SELECT COUNT(*) FROM negative_pairs
    """).fetchone()[0]

    print(f"Negative examples: {negative_count:,}")

    # ---------------------------------------------------------
    # Combine
    # ---------------------------------------------------------

    print("\n[4/6] Combining samples...")

    con.execute("""
    CREATE OR REPLACE TEMP TABLE train_pairs AS

    SELECT *
    FROM positive_pairs

    UNION ALL

    SELECT *
    FROM negative_pairs
    """)

    total = con.execute("""
        SELECT COUNT(*) FROM train_pairs
    """).fetchone()[0]

    print(f"Total training pairs: {total:,}")

    # ---------------------------------------------------------
    # Join source data
    # ---------------------------------------------------------

    print("\n[5/6] Joining source data...")

    con.execute("""
    CREATE OR REPLACE TEMP TABLE s1 AS
    SELECT
        entity_id,
        business_name_norm AS s1_name,
        business_name_ascii AS s1_name_ascii,
        business_name_compact AS s1_name_compact,
        business_name_token_sorted AS s1_name_token_sorted,

        business_address_norm AS s1_address,
        business_address_ascii AS s1_address_ascii,
        business_address_token_sorted AS s1_address_token_sorted,

        house_number AS s1_house,
        country_norm AS s1_country

    FROM read_csv(
        'data/normalized/train_source1.tsv',
        delim='\t',
        header=true,
        all_varchar=true
    )
    """)

    con.execute("""
    CREATE OR REPLACE TEMP TABLE targets AS

    SELECT
        entity_id,
        business_name_norm AS target_name,
        business_name_ascii AS target_name_ascii,
        business_name_compact AS target_name_compact,
        business_name_token_sorted AS target_name_token_sorted,

        business_address_norm AS target_address,
        business_address_ascii AS target_address_ascii,
        business_address_token_sorted AS target_address_token_sorted,

        house_number AS target_house,
        country_norm AS target_country

    FROM read_csv(
        'data/normalized/train_source2.tsv',
        delim='\t',
        header=true,
        all_varchar=true
    )

    UNION ALL

    SELECT
        entity_id,
        business_name_norm,
        business_name_ascii,
        business_name_compact,
        business_name_token_sorted,

        business_address_norm,
        business_address_ascii,
        business_address_token_sorted,

        house_number,
        country_norm

    FROM read_csv(
        'data/normalized/train_source3.tsv',
        delim='\t',
        header=true,
        all_varchar=true
    )
    """)

    # ---------------------------------------------------------
    # Create joined sample
    # ---------------------------------------------------------

    con.execute("""
    CREATE OR REPLACE TEMP TABLE joined AS

    SELECT
        p.*,

        s.s1_name,
        s.s1_name_ascii,
        s.s1_name_compact,
        s.s1_name_token_sorted,

        s.s1_address,
        s.s1_address_ascii,
        s.s1_address_token_sorted,

        s.s1_house,
        s.s1_country,

        t.target_name,
        t.target_name_ascii,
        t.target_name_compact,
        t.target_name_token_sorted,

        t.target_address,
        t.target_address_ascii,
        t.target_address_token_sorted,

        t.target_house,
        t.target_country

    FROM train_pairs p

    JOIN s1 s
        ON p.source1_entity_id = s.entity_id

    JOIN targets t
        ON p.candidate_entity_id = t.entity_id
    """)

    # ---------------------------------------------------------
    # Export a temporary joined sample
    # ---------------------------------------------------------

    temp_file = BASE / "output" / "train_pairs_sample.tsv"

    print("\nWriting temporary training sample...")

    con.execute(f"""
    COPY (
        SELECT *
        FROM joined
    )
    TO '{temp_file}'
    WITH (
        FORMAT CSV,
        DELIMITER '\t',
        HEADER TRUE
    )
    """)

    con.close()

    # ---------------------------------------------------------
    # Calculate fuzzy features in chunks
    # ---------------------------------------------------------

    print("\n[6/6] Calculating similarity features...")

    first = True
    processed = 0

    for chunk in pd.read_csv(
        temp_file,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=25_000
    ):

        rows = []

        for row in chunk.to_dict("records"):
            rows.append(similarity_features(row))

        feature_df = pd.DataFrame(rows)

        feature_df.to_csv(
            OUTPUT,
            sep="\t",
            index=False,
            mode="w" if first else "a",
            header=first
        )

        first = False

        processed += len(chunk)

        print(
            f"Features processed: {processed:,}/{total:,}"
        )

    # ---------------------------------------------------------
    # Remove temporary sample
    # ---------------------------------------------------------

    temp_file.unlink()

    print("\n" + "=" * 70)
    print("TRAINING FEATURES COMPLETE")
    print("=" * 70)

    print(f"Output: {OUTPUT}")

    print("\nFeature rows:")

    df = pd.read_csv(
        OUTPUT,
        sep="\t",
        nrows=5
    )

    print(df.columns.tolist())

    print("\nDONE")


if __name__ == "__main__":
    main()
