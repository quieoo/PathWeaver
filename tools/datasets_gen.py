import json
import argparse

# python 2.datasets_gen.py -t musique -p1 /mnt/n0/datasets/MuSiQue/musique_ans_v1.0_train.jsonl -p2 /mnt/n0/datasets/MuSiQue/musique_ans_v1.0_train_triple_llama3_8B_instruct_inst6_num_sample6000.json -p3 ../datasets/musique/train_6000.jsonl

def merge_triples_musique(jsonl_path, triples_json_path, output_path):
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        original_data = [json.loads(line) for line in f if line.strip()]

    with open(triples_json_path, 'r', encoding='utf-8') as f:
        triples_data = json.load(f)

    triple_map = {item["sample_id"]: item for item in triples_data}

    merged = []

    for sample in original_data:
        sample_id = sample["id"]
        if sample_id in triple_map:
            triple_sample = triple_map[sample_id]
            triple_paragraphs = triple_sample["paragraphs"]

            para_triple_map = {p["paragraph_id"]: p.get("triples", []) for p in triple_paragraphs}

            for para in sample["paragraphs"]:
                pid = para["idx"]
                if pid in para_triple_map:
                    para["triples"] = para_triple_map[pid]
                else:
                    para["triples"] = []
            merged.append(sample)
        

    print(f"✅ Final number of samples: {len(merged)}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"💾 Output file saved to: {output_path}")


def main():
    """Main function to process command line arguments and execute operations"""
    parser = argparse.ArgumentParser(description="Generate training datasets")
    parser.add_argument("-t", "--dataset_type", type=str, required=True, 
                        help="Dataset type (e.g., musique)")
    parser.add_argument("-p1", "--dataset_path_1", type=str, required=True, 
                        help="First dataset path (original dataset, JSONL format)")
    parser.add_argument("-p2", "--dataset_path_2", type=str, required=True, 
                        help="Second dataset path (triples data, JSON format)")
    parser.add_argument("-p3", "--dataset_path_3", type=str, required=True, 
                        help="Third dataset path (output path, JSON format)")
    
    args = parser.parse_args()

    if args.dataset_type == "musique":
        merge_triples_musique(args.dataset_path_1, args.dataset_path_2, args.dataset_path_3)
    else:
        print(f"Unsupported dataset type: {args.dataset_type}")
        exit(1)


if __name__ == "__main__":
    main()