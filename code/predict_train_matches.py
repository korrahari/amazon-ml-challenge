import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from lightgbm import Booster


BASE = Path(".")
DB_PATH = BASE / "blocking.duckdb"
MODEL_PATH = BASE / "models" / "entity_matcher.txt"
THRESHOLD_PATH = BASE / "models" / "best_threshold.txt"

OUTPUT = BASE / "output" / "train_predicted_matches.tsv"

BATCH_SIZE = 20_000


# ------------------------------------------------------------
# FEATURE FUNCTION
# ------------------------------------------------------------

def make_features(df):

    rows = []

    for row in df.to_dict("records"):

        name1 = row["s1_name"] or ""
        name2 = row["target_name"] or ""

        addr1 = row["s1_address"] or ""
        addr2 = row["target_address"] or ""

        name1_ascii = row["s1_name_ascii"] or ""
        name2_ascii = row["target_name_ascii"] or ""

        addr1_ascii = row["s1_address_ascii"] or ""
        addr2_ascii = row["target_address_ascii"] or ""

        rows.append({
            "name_ratio":
            fuzz.ratio(name1, name2) / 100.0,

            "name_partial":
            fuzz.partial_ratio(name1, name2) / 100.0,

            "name_token_sort":
            fuzz.token_sort_ratio(name1, name2) / 100.0,

            "name_token_set":
            fuzz.token_set_ratio(name1, name2) / 100.0,

            "name_jaro":
            JaroWinkler.normalized_similarity(
                name1,
                name2
            ),

            "name_ascii_ratio":
            fuzz.ratio(
                name1_ascii,
                name2_ascii
            ) / 100.0,

            "name_exact":
            int(name1 == name2),

            "name_compact_exact":
            int(
                row["s1_name_compact"]
                == row["target_name_compact"]
            ),

            "name_token_exact":
            int(
                row["s1_name_token_sorted"]
                == row["target_name_token_sorted"]
            ),

            "address_ratio":
            fuzz.ratio(addr1, addr2) / 100.0,

            "address_partial":
            fuzz.partial_ratio(addr1, addr2) / 100.0,

            "address_token_sort":
            fuzz.token_sort_ratio(addr1, addr2) / 100.0,

            "address_token_set":
            fuzz.token_set_ratio(addr1, addr2) / 100.0,

            "address_ascii_ratio":
            fuzz.ratio(
                addr1_ascii,
                addr2_ascii
            ) / 100.0,

            "address_exact":
            int(addr1 == addr2),

            "address_token_exact":
            int(
                row["s1_address_token_sorted"]
                == row["target_address_token_sorted"]
            ),

            "country_match":
            int(
                row["s1_country"]
                == row["target_country"]
            ),

            "house_match":
            int(
                row["s1_house"] != ""
                and row["s1_house"]
                == row["target_house"]
            ),

            "name_length_diff":
            abs(len(name1) - len(name2)),

            "address_length_diff":
            abs(len(addr1) - len(addr2)),

            "combined_score":
            (
                0.60 *
                fuzz.ratio(name1, name2) / 100.0
                +
                0.40 *
                fuzz.ratio(addr1, addr2) / 100.0
            ),

            "strong_name_address":
            int(
                fuzz.ratio(name1, name2) / 100.0 >= 0.85
                and
                fuzz.ratio(addr1, addr2) / 100.0 >= 0.70
            ),

            "source_is_s3":
            int(row["source"] == "S3")
        })

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("BATCH PREDICTION ON TRAIN CANDIDATES")
    print("=" * 70)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\n[1/5] Loading model...")

    model = Booster(
        model_file=str(MODEL_PATH)
    )

    with open(THRESHOLD_PATH) as f:
        threshold = float(f.read().strip())

    print(f"Threshold: {threshold}")

    # --------------------------------------------------------
    # Connect DuckDB
    # --------------------------------------------------------

    print("\n[2/5] Connecting to candidate database...")

    con = duckdb.connect(str(DB_PATH))

    con.execute("PRAGMA memory_limit='7GB'")
    con.execute("PRAGMA threads=2")
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute("PRAGMA temp_directory='duckdb_tmp'")
    con.execute("PRAGMA max_temp_directory_size='6GB'")

    total = con.execute("""
        SELECT COUNT(*)
        FROM candidates
    """).fetchone()[0]

    print(f"Total candidate pairs: {total:,}")
    print(f"Batch size: {BATCH_SIZE:,}")

    # --------------------------------------------------------
    # Create output
    # --------------------------------------------------------

    output_path = str(OUTPUT)

    Path(output_path).unlink(missing_ok=True)

    first_write = True

    processed = 0
    matched = 0
    first_write = True

    min_rowid = con.execute(
        "SELECT MIN(rowid) FROM candidates"
    ).fetchone()[0]

    max_rowid = con.execute(
        "SELECT MAX(rowid) FROM candidates"
    ).fetchone()[0]

    current_rowid = min_rowid

    while current_rowid <= max_rowid:

        end_rowid = current_rowid + BATCH_SIZE

        query = f"""
        SELECT
            c.source1_entity_id,
            c.candidate_entity_id,
            c.source,

            s.business_name_norm AS s1_name,
            s.business_name_norm AS s1_name_ascii,
            s.business_name_compact AS s1_name_compact,
            s.business_name_token_sorted AS s1_name_token_sorted,

            s.business_address_norm AS s1_address,
            s.business_address_norm AS s1_address_ascii,
            s.business_address_token_sorted AS s1_address_token_sorted,

            s.house_number AS s1_house,
            s.country_norm AS s1_country,

            t.business_name_norm AS target_name,
            t.business_name_norm AS target_name_ascii,
            t.business_name_compact AS target_name_compact,
            t.business_name_token_sorted AS target_name_token_sorted,

            t.business_address_norm AS target_address,
            t.business_address_norm AS target_address_ascii,
            t.business_address_token_sorted AS target_address_token_sorted,

            t.house_number AS target_house,
            t.country_norm AS target_country

        FROM candidates c

        JOIN s1 s
            ON c.source1_entity_id = s.source1_entity_id

        JOIN targets t
            ON c.candidate_entity_id = t.candidate_entity_id
           AND c.source = t.source

        WHERE c.rowid >= {current_rowid}
          AND c.rowid < {end_rowid}
        """

        batch = con.execute(query).fetchdf()

        if len(batch) == 0:
            current_rowid = end_rowid
            continue

        feature_df = make_features(batch)

        probabilities = model.predict(feature_df)

        keep = probabilities >= threshold

        if keep.any():

            selected = batch.loc[
                keep,
                [
                    "source1_entity_id",
                    "candidate_entity_id",
                    "source"
                ]
            ].copy()

            selected["match_probability"] = probabilities[keep]

            selected.to_csv(
                OUTPUT,
                sep="\t",
                index=False,
                mode="w" if first_write else "a",
                header=first_write
            )

            first_write = False
            matched += len(selected)

        processed += len(batch)
        current_rowid = end_rowid

        if processed % 200000 < BATCH_SIZE:
            pct = processed / total * 100
            print(
                f"Processed: {processed:,}/{total:,} "
                f"({pct:.2f}%) | matches: {matched:,}"
            )

    con.close()

    print("\n[4/5] Prediction complete.")

    print(f"Candidates processed: {processed:,}")
    print(f"Predicted matches:    {matched:,}")

    print("\n[5/5] Output:")
    print(OUTPUT)

    print("\n" + "=" * 70)
    print("TRAIN CANDIDATE PREDICTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
