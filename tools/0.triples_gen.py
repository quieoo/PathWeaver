import json
from venv import create
from validate_json_output import fix_triple_extraction_response, validate_output
from validate_json_schema import stage_to_schema
import json_repair
import argparse
from vllm import LLM, SamplingParams
from tqdm import tqdm
from triple_extraction_prompt import TRIPLE_INSTRUCTIONS,stage_to_prompt_type


def create_batch_instructions(batch_data, inst_type):
    batched_instructions=[]
    for item in batch_data:        
        system_msg=TRIPLE_INSTRUCTIONS["en"]['system']
        stage_msg=TRIPLE_INSTRUCTIONS["en"][inst_type]+TRIPLE_INSTRUCTIONS["en"]['passage_start']+'\n'+item

        # Format as a single prompt string for vLLM
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{stage_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        batched_instructions.append(prompt)
    return batched_instructions

def generate_and_format(model, contexts, batch_size, triplet_inst_id=1):
    # Set stop tokens to stop generation when the model finishes outputting JSON
    sampling_params = SamplingParams(
        temperature=0.7, 
        max_tokens=512, 
        stop=["\n\n", "}\n\n", "]", "]\n"]  # Stop at common JSON endings
    )


    total_batches = (len(contexts) + batch_size - 1) // batch_size
    
    batch_iterator = tqdm(range(0, len(contexts), batch_size), total=total_batches, desc="Processing batches")

    structured_outputs=[]
    for i in batch_iterator:

        # create instructions
        batch_paragraphs = contexts[i:i+batch_size]
        # print(f"0.Paragraphs: {batch_paragraphs}")

        # instructions = create_batch_instructions_0(batch_paragraphs)
        instructions = create_batch_instructions(batch_paragraphs, stage_to_prompt_type.get(triplet_inst_id, "entity_relation"))

        # print(f"1.Instructions: {instructions}")

        # run model
        results = model.generate(instructions, sampling_params)
        # print(f"2.Model Results: {results}")
        contents=[]
        # extract content
        for result in results:
            if hasattr(result, 'outputs') and len(result.outputs) > 0:
                contents.append(result.outputs[0].text)
            else:
                contents.append("")
        # print(f"3.Contents: {contents}")
        # run validation
        validate_kwargs = {
            'schema': stage_to_schema.get(triplet_inst_id, None),
            'fix_function': fix_triple_extraction_response,
            'prompt_type': stage_to_prompt_type.get(triplet_inst_id, None),
            'allow_empty': True,
        }
        failed_indices = []
        for i,content in enumerate(contents):
            try:
                contents[i]=validate_output(content, **validate_kwargs)
            except Exception as e:
                print(f"Error validate for index {i}: {e}")
                failed_indices.append(i)
        # skip retrying
        for i in failed_indices:
            contents[i]=""
        # print(f"4.Validate Results: {contents}")
        # parse and extrace structured data
        for content in contents:
            try:
                triples = json_repair.loads(content)
                if isinstance(triples, list):
                    structured_outputs.append(triples)
                else:
                    structured_outputs.append([])
            except Exception as e:
                print(f"[Error] JSON parse failed: {e}")
                structured_outputs.append([])
        
        # print(f"5.Structured Outputs: {structured_outputs}")
    
    # Deduplicate and merge triples within each paragraph
    return structured_outputs


# TODO: Instead of running hard-coded deduplication, enhance the instruction to avoid generating duplicated triples may be better.
def dedup_triples(triples_in_paragraph):
    unique_map = {}
    for t in triples_in_paragraph:
        head = t.get("Head", "").strip()
        rel = t.get("Relation", "").strip()
        tail = t.get("Tail", "").strip()
        key = (head, rel)

        if not head or not rel or not tail:
            continue

        if key not in unique_map:
            unique_map[key] = tail
        else:
            prev_tail = unique_map[key]
            # Case 1: Containment (keep longer)
            if prev_tail in tail:
                unique_map[key] = tail
            elif tail in prev_tail:
                pass  # keep previous (longer) one
            # Case 2: Parallel/Comma-join (merge short distinct values)
            elif len(prev_tail) < 50 and len(tail) < 50:
                # avoid merging duplicates
                merged = set([x.strip() for x in (prev_tail + "," + tail).split(",") if x.strip()])
                unique_map[key] = ", ".join(sorted(merged))
            # Case 3: Conflict (keep first)
            else:
                # optional: print(f"Conflict detected for {key}: {prev_tail} vs {tail}")
                pass
    merged_triples = [
        {"Head": k[0], "Relation": k[1], "Tail": v}
        for k, v in unique_map.items()
    ]

    return merged_triples


def process_musique(llm, data_path, batch_size, model_name, num_sample=-1):
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"Total number of samples: {len(data)}")

    if num_sample > 0:
        data=data[:num_sample]

    all_contexts = []  
    sample_paragraph_counts = []  
    
    # --- Flatten all paragraphs ---
    for sample in data:
        paragraphs = sample['paragraphs']
        sample_paragraph_counts.append(len(paragraphs))
        all_contexts.extend([paragraph['paragraph_text'] for paragraph in paragraphs])

    inst_id=6
    triples_results = generate_and_format(llm, all_contexts, batch_size, inst_id)

    print(f"Total number of triples: {len(triples_results)}")

    output = []
    triple_index = 0
    
    for i, sample in enumerate(data):
        sample_id = sample.get('id', i) 
        paragraphs_data = []

        for j in range(sample_paragraph_counts[i]):
            triples_in_paragraph = triples_results[triple_index] if triple_index < len(triples_results) else []
            triples_in_paragraph=dedup_triples(triples_in_paragraph)

            paragraphs_data.append({
                "paragraph_id": j,
                "paragraph_text": sample["paragraphs"][j]["paragraph_text"],
                "triples": triples_in_paragraph,
            })
            triple_index += 1
            
        output.append({
            "sample_id": sample_id,
            "paragraphs": paragraphs_data
        })
    
    suffix_str=f"_triple_{model_name}_inst{str(inst_id)}_num_sample{str(num_sample)}.json"
    output_path = data_path.replace(".jsonl", suffix_str)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=4, ensure_ascii=False)


# TODO:fix
def process_musique_on_save(
    llm, data_path, batch_size, model_name, num_sample=-1, save_every=1
):
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"Total number of samples: {len(data)}")

    if num_sample > 0:
        data = data[:num_sample]

    # --- flatten all paragraphs ---
    all_contexts = []
    sample_paragraph_counts = []
    for sample in data:
        paragraphs = sample["paragraphs"]
        sample_paragraph_counts.append(len(paragraphs))
        all_contexts.extend([p["paragraph_text"] for p in paragraphs])

    inst_id = 6
    total_batches = (len(all_contexts) + batch_size - 1) // batch_size
    print(f"Total batches to process: {total_batches}")

    output = []
    triple_index = 0
    processed_batches = 0
    accumulated_paragraphs = 0

    for batch_start in tqdm(range(0, len(all_contexts), batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, len(all_contexts))
        batch_contexts = all_contexts[batch_start:batch_end]
        processed_batches += 1

        # --- Generate triples for this batch ---
        triples_results = generate_and_format(llm, batch_contexts, batch_size, inst_id)

        # --- Attach triples back to their samples ---
        for i, sample in enumerate(data):
            print(f"Periodicly printing progress: {processed_batches}/{total_batches}")
            sample_id = sample.get("id", i)
            paragraphs_data = []
            for j in range(sample_paragraph_counts[i]):
                if triple_index < len(triples_results):
                    triples_in_paragraph = triples_results[triple_index]
                    triple_index += 1
                else:
                    triples_in_paragraph = []

                # Deduplicate per paragraph
                before = len(triples_in_paragraph)
                triples_in_paragraph = dedup_triples(triples_in_paragraph)
                after = len(triples_in_paragraph)
                if before != after:
                    print(f"[Dedup] Sample {sample_id}, Paragraph {j}: {before}→{after}")

                paragraphs_data.append(
                    {
                        "paragraph_id": j,
                        "paragraph_text": sample["paragraphs"][j]["paragraph_text"],
                        "triples": triples_in_paragraph,
                    }
                )
            output.append({"sample_id": sample_id, "paragraphs": paragraphs_data})

        accumulated_paragraphs += len(batch_contexts)

        # --- Periodic save ---
        if processed_batches % save_every == 0:
            partial_path = data_path.replace(
                ".json",
                f"_partial_batch{processed_batches}_triple_{model_name}_inst{inst_id}.json",
            )
            with open(partial_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=4, ensure_ascii=False)
            print(f"[Checkpoint Saved] {partial_path} (Processed {accumulated_paragraphs} paragraphs)")

    # --- Final save ---
    suffix_str = f"_triple_{model_name}_inst{inst_id}_num_sample{num_sample}.json"
    output_path = data_path.replace(".json", suffix_str)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print(f"[Final Saved] {output_path}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="extract triples")
    parser.add_argument("--dataset_path", type=str, required=True, help="path to dataset file")
    parser.add_argument("--dataset_type", type=str, required=True, choices=["musique"], help="dataset type")
    parser.add_argument("--model_path", type=str, required=True, help="path of local model")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--num_sample", type=int, default=-1, help="number of samples to process")
    
    args = parser.parse_args()
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        dtype="float16",  
        max_model_len=8192 
    )

    if args.dataset_type == "musique":
        process_musique(llm, args.dataset_path, args.batch_size, args.model_path.split('/')[-1], args.num_sample)
    else:
        print(f"Unsupported dataset type: {args.dataset_type}")

#DEBUG USE
# python 0.triples_gen.py --dataset_path /mnt/n0/datasets/MuSiQue/musique_ans_v1.0_train.jsonl  --dataset_type musique --model_path /mnt/n0/models/llama3_8B_instruct --batch_size 1 --num_sample 1


# nohup python 0.triples_gen.py --dataset_path /mnt/n0/datasets/MuSiQue/musique_ans_v1.0_train.jsonl  --dataset_type musique --model_path /mnt/n0/models/llama3_8B_instruct --batch_size 20 --num_sample 6000 > triple.log 2>&1 &