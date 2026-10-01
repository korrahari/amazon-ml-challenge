import pandas as pd

FILE = "output/matching_results_v2.tsv"

EXPECTED_S1 = "data/normalized/test_source1.tsv"

print("Loading V2...")
sub = pd.read_csv(
    FILE,
    sep="\t",
    dtype=str,
    keep_default_na=False
)

print("Loading test S1...")
s1 = pd.read_csv(
    EXPECTED_S1,
    sep="\t",
    dtype=str,
    usecols=["entity_id"]
)

expected = set(s1["entity_id"])
submitted = set(sub["source1_entity_id"])

print()
print("=" * 50)
print("V2 SUBMISSION CHECK")
print("=" * 50)

print("Columns:", list(sub.columns))
print("Rows:", len(sub))
print("Unique S1:", sub["source1_entity_id"].nunique())
print("Expected S1:", len(expected))

invalid_s1 = submitted - expected

print("Invalid S1 IDs:", len(invalid_s1))

# Check duplicate S1 rows
duplicate_rows = (
    sub["source1_entity_id"].value_counts() > 1
).sum()

print("Duplicate S1 rows:", duplicate_rows)

# Check empty matched IDs
empty = (
    sub["matched_entity_ids"]
    .fillna("")
    .eq("")
).sum()

print("Empty-match rows:", empty)
print("Rows with matches:", len(sub) - empty)

# Check malformed rows
bad_rows = (
    sub["source1_entity_id"].eq("") |
    sub["matched_entity_ids"].isna()
).sum()

print("Malformed rows:", bad_rows)

# Check header
correct_header = list(sub.columns) == [
    "source1_entity_id",
    "matched_entity_ids"
]

print("Correct header:", correct_header)

valid = (
    len(sub) == len(expected)
    and sub["source1_entity_id"].nunique() == len(expected)
    and not invalid_s1
    and duplicate_rows == 0
    and bad_rows == 0
    and correct_header
)

print()

if valid:
    print("V2 STRUCTURE: VALID")
else:
    print("V2 STRUCTURE: INVALID")

print("=" * 50)
