import json

from csv import DictWriter


# !wget https://huggingface.co/datasets/allenai/nllb/raw/main/dataset_infos.json -O nllb_info.json

nllb_info = json.loads(open("nllb_info.json").read())

nllb_sizes = {k: v["splits"]["train"]["num_examples"] for k, v in nllb_info.items()}
nllb_sizes = {k: v for k, v in nllb_sizes.items() if v > 0}
nllb_sizes = [{"lang_code": k, "size": v} for k, v in nllb_sizes.items()]


writer = DictWriter(open("nllb_sizes.csv", "w"), ["lang_code", "size"])
writer.writeheader()
writer.writerows(nllb_sizes)
