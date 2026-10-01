import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

files = [
    "train_source1.tsv",
    "train_source2.tsv",
    "train_source3.tsv",
    "train_ground_truth.tsv",
    "test_source1.tsv",
    "test_source2.tsv",
    "test_source3.tsv",
]

for filename in files:
    path = DATA_DIR / filename

    print("\n" + "=" * 70)
    print(filename)
    print("=" * 70)

    if not path.exists():
        print("FILE NOT FOUND")
        continue

    df = pd.read_csv(path, sep="\t", nrows=5)

    print("Columns:")
    print(list(df.columns))

    print("\nFirst 5 rows:")
    print(df.to_string(index=False))