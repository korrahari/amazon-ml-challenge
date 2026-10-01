import pandas as pd
import re
from rapidfuzz.fuzz import ratio, WRatio

MISSED = "output/v2_missed_s1.tsv"
S1_FILE = "data/normalized/train_source1.tsv"

TARGET_FILES = [
    "data/normalized/train_source2.tsv",
    "data/normalized/train_source3.tsv"
]

OUT = "output/v2_stream_recovered.tsv"


def tokens(text):
    return {
        x for x in str(text).split()
        if len(x) >= 4
    }


def numbers(text):
    return set(re.findall(r"\b\d+\b", str(text)))


# -------------------------------------------------
# Load difficult S1 records
# -------------------------------------------------

missed = pd.read_csv(MISSED, sep="\t")

missed_ids = set(missed.source1_entity_id)

s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    usecols=[
        "entity_id",
        "country_norm",
        "business_name_norm",
        "business_address_norm"
    ]
)

s1 = s1[s1.entity_id.isin(missed_ids)].copy()

print("Difficult S1:", len(s1))


# -------------------------------------------------
# Build lookup indexes
# -------------------------------------------------

number_index = {}
token_index = {}

s1_records = {}

for _, r in s1.iterrows():

    sid = r.entity_id
    country = str(r.country_norm)

    s1_records[sid] = (
        str(r.business_name_norm),
        str(r.business_address_norm)
    )

    nums = numbers(r.business_address_norm)
    toks = tokens(r.business_address_norm)

    for n in nums:

        key = (country, n)

        if key not in number_index:
            number_index[key] = set()

        number_index[key].add(sid)

    for t in toks:

        key = (country, t[:5])

        if key not in token_index:
            token_index[key] = set()

        token_index[key].add(sid)


print("Number keys:", len(number_index))
print("Token keys:", len(token_index))
print("S1 records indexed:", len(s1_records))


# -------------------------------------------------
# Stream target files
# -------------------------------------------------

recovered = {}

for target_file in TARGET_FILES:

    print("\nScanning:", target_file)

    for chunk in pd.read_csv(
        target_file,
        sep="\t",
        usecols=[
            "entity_id",
            "country_norm",
            "business_name_norm",
            "business_address_norm"
        ],
        chunksize=50000
    ):

        for _, r in chunk.iterrows():

            tid = r.entity_id
            country = str(r.country_norm)

            addr = str(r.business_address_norm)
            name = str(r.business_name_norm)

            nums = numbers(addr)
            toks = tokens(addr)

            candidate_s1 = set()

            # Address-number candidates
            for n in nums:

                candidate_s1.update(
                    number_index.get(
                        (country, n),
                        set()
                    )
                )

            # Address-token candidates
            for t in toks:

                candidate_s1.update(
                    token_index.get(
                        (country, t[:5]),
                        set()
                    )
                )

            if not candidate_s1:
                continue

            for sid in candidate_s1:

                s1_name, s1_addr = s1_records[sid]

                name_score = WRatio(
                    s1_name,
                    name
                )

                address_score = ratio(
                    s1_addr,
                    addr
                )

                number_overlap = bool(
                    numbers(s1_addr) & nums
                )

                if (
                    number_overlap
                    and name_score >= 70
                    and address_score >= 45
                ) or (
                    number_overlap
                    and name_score >= 85
                ):

                    recovered.setdefault(
                        sid,
                        set()
                    ).add(tid)

        print(
            "Recovered S1 so far:",
            len(recovered)
        )


# -------------------------------------------------
# Write recovered matches only
# -------------------------------------------------

with open(OUT, "w") as f:

    f.write(
        "source1_entity_id\t"
        "matched_entity_ids\n"
    )

    for sid, tids in recovered.items():

        f.write(
            sid + "\t" +
            ",".join(sorted(tids)) +
            "\n"
        )


print("\nDONE")
print("Recovered S1:", len(recovered))
print("Output:", OUT)

