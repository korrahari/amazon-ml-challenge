import re
import unicodedata
from unidecode import unidecode

LEGAL = {
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

def normalize_unicode(value):
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value).strip()

    words = value.split()
    words = [LEGAL.get(w, w) for w in words]

    return " ".join(words)

def normalize_ascii(value):
    value = unidecode(value).casefold()
    value = re.sub(r"[^\w\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value

def token_sorted(value):
    return " ".join(sorted(set(value.split())))

def compact(value):
    return re.sub(r"[^a-z0-9]", "", value)

samples = [
    "राम मार्केटिंग प्राइवेट लिमिटेड",
    "સધર્ન ઇન્ફોટેક પ્રાઇવેટ લિમિટેડ",
    "परफेक्ट बिजनेस प्राइवेट लिमिटेड",
    "Innovative Technology",
    "Middleton Alussa Clinic",
    "Alussa Middleton Clinic",
    "LLC Allied Fund",
    "Allied Fund LLC",
    "L0la D. Alexander, P.A.",
    "Basalt Inc | www.basaltinc.com",
]

for value in samples:
    unicode_name = normalize_unicode(value)
    ascii_name = normalize_ascii(unicode_name)
    sorted_name = token_sorted(ascii_name)
    compact_name = compact(ascii_name)

    print("=" * 80)
    print("ORIGINAL      :", value)
    print("UNICODE       :", unicode_name)
    print("ASCII         :", ascii_name)
    print("TOKEN SORTED  :", sorted_name)
    print("COMPACT       :", compact_name)
