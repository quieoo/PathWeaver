import json

train_data_file="AT2QA_2wiki_train_2hop.json"
output_file="ATFB_2wiki_train_2hop_silver.json"


# train_data_file="AT2QA_2wiki_test_2hop.json"
# output_file="ATFB_2wiki_test_2hop_silver.json"



with open(train_data_file,"r") as f:
    data=json.load(f)

exact_match_cnt=0
contain_count=0
for item in data:
    answer=item["A"]
    triple_lists=item["triple_lists"]
    exact_match_path_id=[]
    contain_path_id=[]
    for i, triple_list in enumerate(triple_lists):
        triple_2=triple_list[1]
        if triple_2["description"]==answer:
            exact_match_path_id.append(i)
        elif answer in triple_2["description"]:
            contain_path_id.append(i)

    if len(exact_match_path_id) > 0:
        exact_match_cnt+=1
        siver_path=triple_lists[exact_match_path_id[0]]
        siver_path_id=exact_match_path_id[0]
    elif len(contain_path_id) > 0:
        contain_count+=1
        siver_path=triple_lists[contain_path_id[0]]
        siver_path_id=contain_path_id[0]
    else:
        siver_path=None
    
    if siver_path is not None:
        triple_lists[siver_path_id]=triple_lists[0]
        triple_lists[0]=siver_path

with open(output_file,"w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Extract silver path from {train_data_file} to {output_file}")
print(f"Total sample: {len(data)}")
print(f"Exact match ratio: {exact_match_cnt/len(data)}")
print(f"Contain ratio: {(contain_count+exact_match_cnt)/len(data)}")
