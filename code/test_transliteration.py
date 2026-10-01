from unidecode import unidecode

samples = [
    "राम मार्केटिंग प्राइवेट लिमिटेड",
    "આદિત્ય પ્રોપર્ટીઝ એલએલપી",
    "સધર્ન ઇન્ફોટેક પ્રાઇવેટ લિમિટેડ",
    "परफेक्ट बिजनेस प्राइवेट लिमिटेड",
    "इनोवेटिव टेक्नोलॉजी",
    "फॉर्च्यून ग्लोबल",
    "リアル モダン フード リミテッド",
]

for value in samples:
    result = unidecode(value)

    print("=" * 70)
    print("ORIGINAL:")
    print(value)
    print()
    print("UNIDECODE:")
    print(result)
