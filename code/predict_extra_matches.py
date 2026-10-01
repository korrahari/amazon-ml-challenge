import duckdb
import lightgbm as lgb
import pandas as pd
from rapidfuzz import fuzz

DB = "blocking.duckdb"
MODEL = "models/entity_matcher.txt"
THRESHOLD = 0.78
OUTPUT = "output/extra_predicted_matches.tsv"
BATCH_SIZE = 20000


def safe(v):
    return "" if v is None else str(v)


def make_features(df):
    rows = []

    for _, row in df.iterrows():

        name1 = safe(row["s1_name"])
        name2 = safe(row["target_name"])

        name1_ascii = safe(row["s1_name_ascii"])
        name2_ascii = safe(row["target_name_ascii"])

        addr1 = safe(row["s1_address"])
        addr2 = safe(row["target_address"])

        addr1_ascii = safe(row["s1_address_ascii"])
        addr2_ascii = safe(row["target_address_ascii"])

        name_compact1 = safe(row["s1_name_compact"])
        name_compact2 = safe(row["target_name_compact"])

        name_token1 = safe(row["s1_name_token_sorted"])
        name_token2 = safe(row["target_name_token_sorted"])

        addr_token1 = safe(row["s1_address_token_sorted"])
        addr_token2 = safe(row["target_address_token_sorted"])

        house1 = safe(row["s1_house"])
        house2 = safe(row["target_house"])

        country1 = safe(row["s1_country"])
        country2 = safe(row["target_country"])

        name_ratio = fuzz.ratio(name1, name2) / 100
        name_partial = fuzz.partial_ratio(name1, name2) / 100
        name_token_sort = fuzz.token_sort_ratio(name1, name2) / 100
        name_token_set = fuzz.token_set_ratio(name1, name2) / 100
        name_jaro = fuzz.WRatio(name1, name2) / 100
        name_ascii_ratio = fuzz.ratio(name1_ascii, name2_ascii) / 100

        address_ratio = fuzz.ratio(addr1, addr2) / 100
        address_partial = fuzz.partial_ratio(addr1, addr2) / 100
        address_token_sort = fuzz.token_sort_ratio(addr1, addr2) / 100
        address_token_set = fuzz.token_set_ratio(addr1, addr2) / 100
        address_ascii_ratio = fuzz.ratio(addr1_ascii, addr2_ascii) / 100

        combined_score = (
            0.55 * name_token_set +
            0.45 * address_token_set
        )

        strong_name_address = (
            1 if name_token_set >= 0.90 and address_token_set >= 0.70
            else 0
        )

        rows.append({
            "name_ratio": name_ratio,
            "name_partial": name_partial,
            "name_token_sort": name_token_sort,
            "name_token_set": name_token_set,
            "name_jaro": name_jaro,
            "name_ascii_ratio": name_ascii_ratio,
            "name_exact": int(name1 == name2 and name1 != ""),
            "name_compact_exact": int(
                name_compact1 == name_compact2 and name_compact1 != ""
            ),
            "name_token_exact": int(
                name_token1 == name_token2 and name_token1 != ""
            ),
            "address_ratio": address_ratio,
            "address_partial": address_partial,
            "address_token_sort": address_token_sort,
            "address_token_set": address_token_set,
            "address_ascii_ratio": address_ascii_ratio,
            "address_exact": int(addr1 == addr2 and addr1 != ""),
            "address_token_exact": int(
                addr_token1 == addr_token2 and addr_token1 != ""
            ),
            "country_match": int(
                country1 == country2 and country1 != ""
            ),
            "house_match": int(
                house1 == house2 and house1 != ""
            ),
            "name_length_diff": abs(len(name1) - len(name2)),
            "address_length_diff": abs(len(addr1) - len(addr2)),
            "combined_score": combined_score,
            "strong_name_address": strong_name_address,
            "source_is_s3": int(row["source"] == "S3")
        })

    return pd.DataFrame(rows)


def main():

    print("=" * 70)
    print("PREDICTING ADDITIONAL BLOCK")
    print("=" * 70)

    print("\n[1/4] Loading model...")

    model = lgb.Booster(model_file=MODEL)

    print("Threshold:", THRESHOLD)

    print("\n[2/4] Connecting to database...")

    con = duckdb.connect(DB)

    total = con.execute("""
        SELECT COUNT(*)
        FROM extra_candidates e
        WHERE NOT EXISTS (
            SELECT 1
            FROM candidates c
            WHERE c.source1_entity_id = e.source1_entity_id
              AND c.candidate_entity_id = e.candidate_entity_id
              AND c.source = e.source
        )
    """).fetchone()[0]

    print("New candidates to score:", f"{total:,}")
    print("Batch size:", BATCH_SIZE)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE extra_new AS
        SELECT
            e.*,
            ROW_NUMBER() OVER () - 1 AS rn
        FROM extra_candidates e
        WHERE NOT EXISTS (
            SELECT 1
            FROM candidates c
            WHERE c.source1_entity_id = e.source1_entity_id
              AND c.candidate_entity_id = e.candidate_entity_id
              AND c.source = e.source
        )
    """)

    output = OUTPUT

    processed = 0
    matched = 0
    first_write = True

    while processed < total:

        batch = con.execute(f"""
            SELECT
                e.source1_entity_id,
                e.candidate_entity_id,
                e.source,

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

            FROM extra_new e

            JOIN s1 s
              ON e.source1_entity_id = s.source1_entity_id

            JOIN targets t
              ON e.candidate_entity_id = t.candidate_entity_id
             AND e.source = t.source

            WHERE e.rn >= {processed}
              AND e.rn < {processed + BATCH_SIZE}
        """).fetchdf()

        if len(batch) == 0:
            break

        features = make_features(batch)

        probabilities = model.predict(features)

        keep = probabilities >= THRESHOLD

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
                output,
                sep="\t",
                index=False,
                mode="w" if first_write else "a",
                header=first_write
            )

            first_write = False
            matched += len(selected)

        processed += len(batch)

        if processed % 200000 < BATCH_SIZE:
            print(
                f"Processed: {processed:,}/{total:,} "
                f"({processed / total * 100:.2f}%) | "
                f"new matches: {matched:,}"
            )

    print("\n[4/4] Complete.")
    print("Candidates processed:", f"{processed:,}")
    print("New predicted matches:", f"{matched:,}")
    print("Output:", output)

    con.close()


if __name__ == "__main__":
    main()
