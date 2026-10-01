import pandas as pd
import re
import unicodedata
from pathlib import Path
from unidecode import unidecode

DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "normalized"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    "train_source1.tsv",
    "train_source2.tsv",
    "train_source3.tsv",
    "test_source1.tsv",
    "test_source2.tsv",
    "test_source3.tsv",
]

LEGAL_REPLACEMENTS = {
    "incorporated": "inc",
    "inc.": "inc",
    "limited": "ltd",
    "ltd.": "ltd",
    "corporation": "corp",
    "corp.": "corp",
    "company": "co",
    "co.": "co",
    "llc.": "llc",
    "private limited": "pvt ltd",
    "private ltd": "pvt ltd",
    "pvt. ltd.": "pvt ltd",
    "pvt ltd.": "pvt ltd",
}

ADDRESS_REPLACEMENTS = {
    "street": "st",
    "st.": "st",
    "road": "rd",
    "rd.": "rd",
    "avenue": "ave",
    "ave.": "ave",
    "boulevard": "blvd",
    "blvd.": "blvd",
    "drive": "dr",
    "dr.": "dr",
    "lane": "ln",
    "ln.": "ln",
    "highway": "hwy",
    "hwy.": "hwy",
    "parkway": "pkwy",
    "place": "pl",
    "court": "ct",
}


def preserve_unicode(value):
    """
    Normalize text while preserving Unicode letters/numbers/combining marks.
    This is important for Hindi, Gujarati, Bengali, Telugu, Japanese, etc.
    """
    if pd.isna(value):
        return ""

    value = str(value)

    # Unicode compatibility normalization
    value = unicodedata.normalize("NFKC", value)

    # Case normalization
    value = value.casefold()

    cleaned = []

    for ch in value:
        category = unicodedata.category(ch)

        # Keep:
        # L = Letter
        # N = Number
        # M = Combining mark
        # whitespace
        if category[0] in ("L", "N", "M") or ch.isspace():
            cleaned.append(ch)
        else:
            cleaned.append(" ")

    value = "".join(cleaned)

    # Collapse spaces
    value = re.sub(r"\s+", " ", value).strip()

    return value


def normalize_name(value):
    value = preserve_unicode(value)

    words = value.split()

    normalized_words = []

    for word in words:
        normalized_words.append(
            LEGAL_REPLACEMENTS.get(word, word)
        )

    return " ".join(normalized_words)


def normalize_address(value):
    value = preserve_unicode(value)

    words = value.split()

    normalized_words = []

    for word in words:
        normalized_words.append(
            ADDRESS_REPLACEMENTS.get(word, word)
        )

    return " ".join(normalized_words)


def ascii_normalize(value):
    """
    Secondary transliteration representation.

    IMPORTANT:
    This is NOT used as the primary representation because
    transliteration can be noisy for Indian/Asian scripts.
    """

    if not value:
        return ""

    value = unidecode(value)

    value = value.casefold()

    value = re.sub(r"[^\w\s]", " ", value)

    value = re.sub(r"\s+", " ", value).strip()

    return value


def compact(value):
    """
    Removes spaces from normalized text.
    Useful for detecting spacing differences.
    """

    if not value:
        return ""

    return re.sub(r"\s+", "", value)


def token_sorted(value):
    """
    Sorts words alphabetically.

    Useful for:
    Middleton Alussa Clinic
    Alussa Middleton Clinic

    Both become:
    alussa clinic middleton
    """

    if not value:
        return ""

    return " ".join(sorted(value.split()))


def extract_house_number(value):
    """
    Extract the first house/building number from an address.
    """

    if not value:
        return ""

    match = re.search(
        r"\b\d+[A-Za-z]?\b",
        value
    )

    if match:
        return match.group(0).casefold()

    return ""


def process_file(filename):

    input_path = DATA_DIR / filename
    output_path = OUTPUT_DIR / filename

    first_chunk = True
    total_rows = 0

    for chunk in pd.read_csv(
        input_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=100_000
    ):

        # -------------------------
        # NAME
        # -------------------------

        chunk["business_name_norm"] = (
            chunk["business_name"]
            .map(normalize_name)
        )

        chunk["business_name_ascii"] = (
            chunk["business_name_norm"]
            .map(ascii_normalize)
        )

        chunk["business_name_compact"] = (
            chunk["business_name_norm"]
            .map(compact)
        )

        chunk["business_name_token_sorted"] = (
            chunk["business_name_norm"]
            .map(token_sorted)
        )

        # -------------------------
        # ADDRESS
        # -------------------------

        chunk["business_address_norm"] = (
            chunk["business_address"]
            .map(normalize_address)
        )

        chunk["business_address_ascii"] = (
            chunk["business_address_norm"]
            .map(ascii_normalize)
        )

        chunk["business_address_compact"] = (
            chunk["business_address_norm"]
            .map(compact)
        )

        chunk["business_address_token_sorted"] = (
            chunk["business_address_norm"]
            .map(token_sorted)
        )

        chunk["house_number"] = (
            chunk["business_address_norm"]
            .map(extract_house_number)
        )

        # -------------------------
        # COUNTRY
        # -------------------------

        chunk["country_norm"] = (
            chunk["country"]
            .map(preserve_unicode)
        )

        # -------------------------
        # WRITE
        # -------------------------

        chunk.to_csv(
            output_path,
            sep="\t",
            index=False,
            mode="w" if first_chunk else "a",
            header=first_chunk
        )

        first_chunk = False
        total_rows += len(chunk)

        print(
            f"{filename}: processed {total_rows:,} rows"
        )

    print(
        f"DONE: {output_path} | "
        f"rows={total_rows:,}"
    )


def main():

    print("=" * 70)
    print("UNICODE-SAFE NORMALIZATION")
    print("=" * 70)

    for filename in FILES:
        process_file(filename)

    print("=" * 70)
    print("NORMALIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
