import pandas as pd

MISSED = "output/v2_missed_s1.tsv"
S1_FILE = "data/normalized/train_source1.tsv"

TARGET_FILES = [
    "data/normalized/train_source2.tsv",
    "data/normalized/train_source3.tsv"
]

OUT = "output/v2_address_candidates.tsv"


def address_keys(address):
    tokens = str(address).split()

    numbers = {x for x in tokens if x.isdigit()}

    words = {
        x[:4]
        for x in tokens
        if len(x) >= 4 and not x.isdigit()
    }

    return numbers, words


# Build index of difficult S1 records
missed = pd.read_csv(MISSED, sep="\t")

s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    usecols=[
        "entity_id",
        "country_norm",
        "business_address_norm"
    ]
)

s1 = s1[s1.entity_id.isin(set(missed.source1_entity_id))]

s1_index = {}

for _, r in s1.iterrows():

    numbers, words = address_keys(
        r.business_address_norm
    )

    country = str(r.country_norm)
    sid = r.entity_id

    for number in numbers:
        for word in words:

            key = (country, number, word)

            if key not in s1_index:
                s1_index[key] = []

            s1_index[key].append(sid)

print("Difficult S1:", len(s1))
print("Relevant address keys:", len(s1_index))


# Stream target files
total = 0

with open(OUT, "w") as f:

    f.write("source1_entity_id\ttarget_entity_id\n")

    for target_file in TARGET_FILES:

        print("\nScanning:", target_file)

        for chunk in pd.read_csv(
            target_file,
            sep="\t",
            usecols=[
                "entity_id",
                "country_norm",
                "business_address_norm"
            ],
            chunksize=100000
        ):

            for _, r in chunk.iterrows():

                numbers, words = address_keys(
                    r.business_address_norm
                )

                country = str(r.country_norm)
                tid = r.entity_id

                matched = set()

                for number in numbers:
                    for word in words:

                        key = (country, number, word)

                        ids = s1_index.get(key)

                        if ids:
                            matched.update(ids)

                for sid in matched:

                    f.write(
                        f"{sid}\t{tid}\n"
                    )

                    total += 1

        print("Candidates so far:", total)

print("\nDONE")
print("Total candidate rows:", total)
print("Output:", OUT)
