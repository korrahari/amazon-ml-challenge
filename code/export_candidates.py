import duckdb
from pathlib import Path

DB_PATH = "blocking.duckdb"
OUTPUT_DIR = Path("output")

OUTPUT_DIR.mkdir(exist_ok=True)

RAW_FILE = OUTPUT_DIR / "train_candidate_pairs_raw.tsv"

print("=" * 70)
print("EXPORTING EXISTING CANDIDATES")
print("=" * 70)

con = duckdb.connect(DB_PATH, read_only=True)

con.execute("SET memory_limit='6GB'")
con.execute("SET threads=2")
con.execute("SET preserve_insertion_order=false")
con.execute("SET temp_directory='duckdb_tmp'")
con.execute("SET max_temp_directory_size='8GB'")

print("\nExporting raw candidate pairs...")

con.execute(f"""
    COPY (
        SELECT
            source1_entity_id,
            candidate_entity_id
        FROM candidates
        ORDER BY source1_entity_id, candidate_entity_id
    )
    TO '{RAW_FILE}'
    (
        HEADER,
        DELIMITER '\\t'
    )
""")

count = con.execute("""
    SELECT COUNT(*)
    FROM candidates
""").fetchone()[0]

print("\nCandidate pairs exported:", count)

con.close()

print("\nOutput:")
print(RAW_FILE)

print("\nDONE.")