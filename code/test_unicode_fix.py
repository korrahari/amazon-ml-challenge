import re
import unicodedata

def preserve_unicode(value):
    value = unicodedata.normalize("NFKC", value)
    value = value.casefold()

    cleaned = []

    for ch in value:
        category = unicodedata.category(ch)

        # Keep:
        # L = letters
        # N = numbers
        # M = combining marks
        # whitespace
        if category[0] in ("L", "N", "M") or ch.isspace():
            cleaned.append(ch)
        else:
            cleaned.append(" ")

    value = "".join(cleaned)
    value = re.sub(r"\s+", " ", value).strip()

    return value


samples = [
    "राम मार्केटिंग प्राइवेट लिमिटेड",
    "સધર્ન ઇન્ફોટેક પ્રાઇવેટ લિમિટેડ",
    "પરફેક્ટ બિઝનેસ પ્રાઇવેટ લિમિટેડ",
    "ইনোভেটিভ টেকনোলজি",
    "ఫార్చ్యూన్ గ్లోబల్",
    "Innovative Technology",
    "Middleton Alussa Clinic",
    "Alussa Middleton Clinic",
]

for value in samples:
    print("=" * 80)
    print("ORIGINAL:")
    print(value)
    print()
    print("FIXED:")
    print(preserve_unicode(value))
