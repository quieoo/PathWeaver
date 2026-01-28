import json
from tqdm import tqdm
import random
# # AT2QAfile
input_file_1= "/mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_train_2hop_compositional.json"
# pure QA file
input_file_2= "/mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/2wiki/2wiki_train_datasets.json"
output_file= "/mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_train_2hop_compositional_gold.json"


# input_file_1="/mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_dev_2hop_compositional_top16.json"
# input_file_2="/mnt/n0/datasets/wiki_hotspot_musique/merged_data/filtered_data/2wiki/2wiki_test_datasets.json"
# output_file= "/mnt/n0/datasets/wiki_hotspot_musique/merged_data/all_triples/AT2QA_2wiki_test_2hop_compositional_gold.json"


dataset1=json.load(open(input_file_1,"r"))
dataset2=json.load(open(input_file_2,"r"))
print(f"Load {input_file_1} with {len(dataset1)} samples")
print(f"Load {input_file_2} with {len(dataset2)} samples")

# 构建一个字典, key-Q, value-item
dataset2_dict={item["id"]:item for item in dataset2}


new_dataset1=[]
found_id_count=0
for data in tqdm(dataset1):
    id=data["id"]
    if id in dataset2_dict:
        data["gold_path"]=dataset2_dict[id]["triple_lists"]
        data["gold_Q"]=dataset2_dict[id]["Q"]
        new_dataset1.append(data)
        found_id_count+=1
print(f"Found {found_id_count} samples in {input_file_2}")

if found_id_count!=len(dataset1):
    print(f"Warning: {len(dataset1)-found_id_count} samples in {input_file_1} not found in {input_file_2}")
    raise ValueError(f"Not all samples in {input_file_1} have gold path in {input_file_2}")

print(f"Merged {len(new_dataset1)} samples from {input_file_1} and {input_file_2}")


random_id=random.randint(0,len(new_dataset1))
print(new_dataset1[random_id])
    

with open(output_file,"w") as f:
    json.dump(new_dataset1,f,indent=2,ensure_ascii=False)

print(f"Extract gold path from {input_file_2} -> {input_file_1}, result saved to {output_file}")