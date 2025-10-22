import argparse
import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="text-embedding-3-large",
        choices=["all-MiniLM-L6-v2", "text-embedding-3-large", "ada-embeddings", "text-embedding-v4"],
    )
    parser.add_argument("--dataset_type", type=str, default="synthetic_data")
    parser.add_argument("--endpoint_url", type=str)
    parser.add_argument("--api_key", type=str)
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset in JSON format.",
    )

    args = parser.parse_args()
    return args


def compute_embeddings(
    encoder_model_spec, string_list, batch_size: int = 100
) -> np.array:
    """Compute embeddings for the given dataset in batches using the encoder model spec."""
    embeddings = []

    chunks = [
        string_list[i : i + batch_size]
        for i in range(0, len(string_list), batch_size)
    ]

    model = SentenceTransformer(encoder_model_spec, device="cuda")
    for chunk in tqdm(chunks):
        embd = model.encode(chunk, convert_to_numpy=True)
        embeddings.append(embd)

    embeddings = np.concatenate(embeddings, 0)
    assert len(embeddings) == len(string_list)
    return embeddings

if __name__ == "__main__":
    args = parser_args()
    key_strings = []
    value_strings = []
    
    if args.dataset_type == "multi_wiki_qa_train":
        sid=0
        reformatted_data = []

        with open(args.dataset_path, "r", encoding="utf-8") as f:
            dataset = [json.loads(line.strip()) for line in f]
            for doc in dataset:

                # collect triples for KV embeddings
                for triple in doc["triples"]:
                    key_strings.append(triple["key_string"])
                    value_strings.append(triple["description"])
                
                # add start_id and num_triples to the old dataset
                reformatted_data.append(json.dumps({
                    **doc,
                    "start_id": sid,
                    "num_triples": len(doc["triples"])
                }))
                sid+=len(doc["triples"])

        with open(args.dataset_path, "w", encoding="utf-8") as f:
            f.write("\n".join(reformatted_data))
    elif args.dataset_type == "musique":
        sid=0
        reformatted_data = []

        with open(args.dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
            for sample in dataset:
                reformatted_paragraphs = []
                for paragraph in sample["paragraphs"]:
                    num_triples = len(paragraph["triples"])

                    for triple in paragraph["triples"]:
                        key_strings.append(f"The {triple['Relation']} of {triple['Head']} is")
                        value_strings.append(triple['Tail'])


                    reformatted_paragraphs.append({
                        **paragraph,
                        "start_id": sid,
                        "num_triples": num_triples
                    })
                    sid += num_triples 

                reformatted_data.append({
                    "id": sample["id"],
                    "paragraphs": reformatted_paragraphs,
                    "question": sample["question"],
                    "question_decomposition": sample["question_decomposition"],
                    "answer": sample["answer"],
                    "answer_aliases": sample["answer_aliases"],
                    "answerable": sample["answerable"],
                })
        with open(args.dataset_path, 'w') as f:
            json.dump(reformatted_data, f, indent=4, ensure_ascii=False)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")                
    
    print(f"Computing embeddings for {len(key_strings)} entities using {args.model_name}")
    if args.model_name == "all-MiniLM-L6-v2":
        key_embeds = compute_embeddings(args.model_name, key_strings)
        value_embeds = compute_embeddings(args.model_name, value_strings)
    else:
        raise NotImplementedError(f"Embedding model '{args.model_name}' is not implemented yet.")

    if args.model_name == "all-MiniLM-L6-v2":
        save_name = "all-MiniLM-L6-v2"
    elif args.model_name == "ada-embeddings":
        save_name = "OAI"
    elif args.model_name == "text-embedding-v4":
        save_name = "text-embedding-v4"
    else:
        save_name = "BigOAI"

    output_dir = os.path.dirname(args.dataset_path) or "."
    base_name = os.path.basename(args.dataset_path).rsplit(".", 1)[0]  

    key_path = os.path.join(output_dir, f"{base_name}_{save_name}_embd_key.npy")
    value_path = os.path.join(output_dir, f"{base_name}_{save_name}_embd_value.npy")

    np.save(key_path, np.array(key_embeds))
    np.save(value_path, np.array(value_embeds))

    print(f"Saved embeddings to {key_path} and {value_path}")
    