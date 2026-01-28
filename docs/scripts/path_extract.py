import json

input_file = "/mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_dev_2hop_compositional.json"

with open(input_file, "r") as f:
    data = json.load(f)

for sample in data:
    sample["triple_lists"] = sample["triple_lists"][0]

with open(input_file, "w") as f:
    json.dump(data, f, indent=2)