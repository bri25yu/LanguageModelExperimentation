from csv import DictReader, DictWriter

import langcodes


inputs = DictReader(open("NLLB and mC4 stats - Sheet1.csv"))  # From https://www.tensorflow.org/datasets/catalog/c4#c4multilingual

writer = DictWriter(open("mc4_sizes.csv", "w"), ["lang_code", "size"])
writer.writeheader()
for d in inputs:
    if "validation" in d["Lang code"]: continue

    lang_code, size = d["Lang code"], d["Number of examples"]
    lang_code = lang_code.replace("'", "")
    lang_code = langcodes.get(lang_code).to_alpha3()

    writer.writerow({"lang_code": lang_code, "size": size})
