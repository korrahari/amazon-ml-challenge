import pandas as pd
import numpy as np
from pathlib import Path
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from lightgbm import Booster

CANDIDATES = Path("output/v3_freq5_candidates.tsv")
MODEL_PATH = Path("models/entity_matcher.txt")
THRESHOLD_PATH = Path("models/best_threshold.txt")

OUT = Path("output/v3_freq5_predicted_matches.tsv")

BATCH_SIZE = 20000

FEATURES = [
    "name_ratio",
    "name_partial",
    "name_token_sort",
    "name_token_set",
    "name_jaro",
    "name_ascii_ratio",
    "name_exact",
    "name_compact_exact",
    "name_token_exact",
    "address_ratio",
    "address_partial",
    "address_token_sort",
    "address_token_set",
    "address_ascii_ratio",
    "address_exact",
    "address_token_exact",
    "country_match",
    "house_match",
    "name_length_diff",
    "address_length_diff",
    "combined_score",
    "strong_name_address",
    "source_is_s3",
]


def load_normalized(path):
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )


def make_features(batch):
    rows = []

    for row in batch.itertuples(index=False):
        name1 = row.s1_name
        name2 = row.target_name

        addr1 = row.s1_address
        addr2 = row.target_address

        name1_ascii = row.s1_name_ascii
        name2_ascii = row.target_name_ascii

        addr1_ascii = row.s1_address_ascii
        addr2_ascii = row.target_address_ascii

        nr = fuzz.ratio(name1, name2) / 100.0
        ar = fuzz.ratio(addr1, addr2) / 100.0

        rows.append({
            "name_ratio": nr,
            "name_partial": fuzz.partial_ratio(name1, name2) / 100.0,
            "name_token_sort": fuzz.token_sort_ratio(name1, name2) / 100.0,
            "name_token_set": fuzz.token_set_ratio(name1, name2) / 100.0,
            "name_jaro": JaroWinkler.normalized_similarity(name1, name2),
            "name_ascii_ratio": fuzz.ratio(
                name1_ascii, name2_ascii
            ) / 100.0,

            "name_exact": int(name1 == name2),
            "name_compact_exact": int(
                row.s1_name_compact == row.target_name_compact
            ),
            "name_token_exact": int(
                row.s1_name_token_sorted ==
                row.target_name_token_sorted
            ),

            "address_ratio": ar,
            "address_partial": fuzz.partial_ratio(addr1, addr2) / 100.0,
            "address_token_sort": fuzz.token_sort_ratio(
                addr1, addr2
            ) / 100.0,
            "address_token_set": fuzz.token_set_ratio(
                addr1, addr2
            ) / 100.0,
            "address_ascii_ratio": fuzz.ratio(
                addr1_ascii, addr2_ascii
            ) / 100.0,

            "address_exact": int(addr1 == addr2),
            "address_token_exact": int(
                row.s1_address_token_sorted ==
                row.target_address_token_sorted
            ),

            "country_match": int(
                row.s1_country == row.target_country
            ),

            "house_match": int(
                row.s1_house != "" and
                row.s1_house == row.target_house
            ),

            "name_length_diff": abs(
                len(name1) - len(name2)
            ),

            "address_length_diff": abs(
                len(addr1) - len(addr2)
            ),

            "combined_score": (
                0.60 * nr +
                0.40 * ar
            ),

            "strong_name_address": int(
                nr >= 0.85 and ar >= 0.70
            ),

            "source_is_s3": int(row.source == "S3"),
        })

    return pd.DataFrame(rows, columns=FEATURES)


def main():

    print("=" * 70)
    print("V3 RARE-CANDIDATE SCORING")
    print("=" * 70)

    print("\nLoading model...")
    model = Booster(model_file=str(MODEL_PATH))

    threshold = float(
        THRESHOLD_PATH.read_text().strip()
    )

    print("Threshold:", threshold)

    print("\nLoading normalized training data...")

    s1_paths = [
        "data/normalized/train_source1.tsv"
    ]

    target_paths = {
        "S2": "data/normalized/train_source2.tsv",
        "S3": "data/normalized/train_source3.tsv",
    }

    s1 = load_normalized(s1_paths[0])

    s1 = s1.rename(columns={
        "entity_id": "source1_entity_id",
        "business_name_norm": "s1_name",
        "business_name_ascii": "s1_name_ascii",
        "business_name_compact": "s1_name_compact",
        "business_name_token_sorted": "s1_name_token_sorted",
        "business_address_norm": "s1_address",
        "business_address_ascii": "s1_address_ascii",
        "business_address_token_sorted": "s1_address_token_sorted",
        "house_number": "s1_house",
        "country_norm": "s1_country",
    })

    s1 = s1.set_index("source1_entity_id")

    print("S1 rows:", len(s1))

    print("\nLoading candidates...")

    candidates = pd.read_csv(
        CANDIDATES,
        sep="\t",
        dtype=str
    )

    print("Candidates:", len(candidates))

    # --------------------------------------------------------
    # Build target lookup only for candidate IDs
    # --------------------------------------------------------

    needed = {
        "S2": set(
            candidates.loc[
                candidates.source == "S2",
                "candidate_entity_id"
            ]
        ),
        "S3": set(
            candidates.loc[
                candidates.source == "S3",
                "candidate_entity_id"
            ]
        ),
    }

    print("Needed S2 targets:", len(needed["S2"]))
    print("Needed S3 targets:", len(needed["S3"]))

    target_maps = {}

    for source, path in target_paths.items():

        print(f"\nScanning {source}...")

        target_map = {}

        for chunk in pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=50000
        ):

            hit = chunk[
                chunk["entity_id"].isin(needed[source])
            ]

            if len(hit):
                for row in hit.itertuples(index=False):

                    target_map[row.entity_id] = {
                        "target_name": row.business_name_norm,
                        "target_name_ascii": row.business_name_ascii,
                        "target_name_compact": row.business_name_compact,
                        "target_name_token_sorted":
                            row.business_name_token_sorted,

                        "target_address": row.business_address_norm,
                        "target_address_ascii":
                            row.business_address_ascii,
                        "target_address_token_sorted":
                            row.business_address_token_sorted,

                        "target_house": row.house_number,
                        "target_country": row.country_norm,
                    }

        target_maps[source] = target_map

        print("Targets loaded:", len(target_map))

    print("\nScoring candidates...")

    OUT.unlink(missing_ok=True)

    total_scored = 0
    total_matched = 0
    first = True

    for start in range(0, len(candidates), BATCH_SIZE):

        batch = candidates.iloc[
            start:start + BATCH_SIZE
        ].copy()

        rows = []

        for row in batch.itertuples(index=False):

            s = s1.loc[row.source1_entity_id]
            t = target_maps[row.source][row.candidate_entity_id]

            rows.append({
                "source1_entity_id":
                    row.source1_entity_id,
                "candidate_entity_id":
                    row.candidate_entity_id,
                "source":
                    row.source,

                "s1_name":
                    s.s1_name,
                "s1_name_ascii":
                    s.s1_name_ascii,
                "s1_name_compact":
                    s.s1_name_compact,
                "s1_name_token_sorted":
                    s.s1_name_token_sorted,

                "s1_address":
                    s.s1_address,
                "s1_address_ascii":
                    s.s1_address_ascii,
                "s1_address_token_sorted":
                    s.s1_address_token_sorted,

                "s1_house":
                    s.s1_house,
                "s1_country":
                    s.s1_country,

                **t
            })

        feature_input = pd.DataFrame(rows)

        X = make_features(feature_input)

        probabilities = model.predict(X)

        keep = probabilities >= threshold

        if np.any(keep):

            selected = feature_input.loc[
                keep,
                [
                    "source1_entity_id",
                    "candidate_entity_id",
                    "source"
                ]
            ].copy()

            selected["match_probability"] = probabilities[keep]

            selected.to_csv(
                OUT,
                sep="\t",
                index=False,
                mode="w" if first else "a",
                header=first
            )

            first = False
            total_matched += len(selected)

        total_scored += len(batch)

        if total_scored % 100000 < BATCH_SIZE:
            print(
                f"Scored {total_scored:,}/{len(candidates):,} "
                f"| matches {total_matched:,}"
            )

    print()
    print("=" * 70)
    print("V3 RARE-CANDIDATE SCORING COMPLETE")
    print("=" * 70)
    print("Candidates scored:", f"{total_scored:,}")
    print("Predicted matches:", f"{total_matched:,}")
    print("Output:", OUT)
    print("=" * 70)


if __name__ == "__main__":
    main()
