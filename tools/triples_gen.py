import json
from venv import create
from validate_json_output import fix_triple_extraction_response, validate_output
from validate_json_schema import stage_to_schema
import json_repair
import argparse
from vllm import LLM, SamplingParams
from tqdm import tqdm
from triple_extraction_prompt import TRIPLE_INSTRUCTIONS,stage_to_prompt_type
import os, time


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




def process_musique(llm, data_path, batch_size, model_name, num_sample=-1, start_idx=0, save_every=-1):
    """
    改进版：按 save_every 分块生成三元组，每块单独保存，避免长时间运行丢失。
    """
    # --- 1️⃣ 加载数据 ---
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    total_data = len(data)
    print(f"Total number of samples in file: {total_data}")

    # --- 2️⃣ 应用样本范围 ---
    if num_sample > 0:
        data = data[:num_sample]
    data = data[start_idx:]

    total_to_process = len(data)
    print(f"Processing samples from index {start_idx} to {start_idx + total_to_process}")

    # --- 3️⃣ 设定 save_every ---
    if save_every <= 0 or save_every > total_to_process:
        save_every = total_to_process
    print(f"Each chunk will process {save_every} samples before saving.")

    # --- 4️⃣ 分块循环 ---
    for chunk_start in range(0, total_to_process, save_every):
        chunk_end = min(chunk_start + save_every, total_to_process)
        sub_data = data[chunk_start:chunk_end]

        all_contexts = []
        sample_paragraph_counts = []

        # --- 扁平化段落 ---
        for sample in sub_data:
            paragraphs = sample['paragraphs']
            sample_paragraph_counts.append(len(paragraphs))
            all_contexts.extend([p['paragraph_text'] for p in paragraphs])

        inst_id = 6
        print(f"[Chunk {chunk_start}-{chunk_end}] Total paragraphs: {len(all_contexts)}")

        # --- 调用生成 ---
        triples_results = generate_and_format(llm, all_contexts, batch_size, inst_id)
        print(f"Generated triples for {len(triples_results)} paragraphs.")

        # --- 构造输出 ---
        output = []
        triple_index = 0

        for i, sample in enumerate(sub_data):
            sample_uid = sample.get('id', i + start_idx + chunk_start)
            paragraphs_data = []

            for j in range(sample_paragraph_counts[i]):
                triples_in_paragraph = triples_results[triple_index] if triple_index < len(triples_results) else []
                triples_in_paragraph = dedup_triples(triples_in_paragraph)
                paragraphs_data.append({
                    "paragraph_id": j,
                    "paragraph_text": sample["paragraphs"][j]["paragraph_text"],
                    "triples": triples_in_paragraph
                })
                triple_index += 1

            output.append({
                "sample_id": sample_uid,
                "paragraphs": paragraphs_data
            })

        # --- 保存结果 ---
        suffix_str = f"_triple_{model_name}_inst{inst_id}_{chunk_start+start_idx}_{chunk_end+start_idx}.json"
        output_path = data_path.replace(".jsonl", suffix_str)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=4, ensure_ascii=False)

        print(f"✅ Saved chunk {chunk_start}-{chunk_end} → {output_path}")

    print("🎯 All chunks processed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="extract triples")
    parser.add_argument("--dataset_path", type=str, required=True, help="path to dataset file")
    parser.add_argument("--dataset_type", type=str, required=True, choices=["musique"], help="dataset type")
    parser.add_argument("--model_path", type=str, required=True, help="path of local model")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--num_sample", type=int, default=-1, help="number of samples to process")
    parser.add_argument("--start_from", type=int, default=0, help="start from which sample")
    parser.add_argument("--save_every", type=int, default=-1, help="save every n samples")
    args = parser.parse_args()
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        dtype="float16",  
        max_model_len=8192 
    )

    if args.dataset_type == "musique":
        process_musique(llm, args.dataset_path, args.batch_size, args.model_path.split('/')[-1], args.num_sample, args.start_from, args.save_every)
    else:
        print(f"Unsupported dataset type: {args.dataset_type}")

#DEBUG USE
# python 0.triples_gen.py --dataset_path /mnt/n0/datasets/MuSiQue/musique_ans_v1.0_train.jsonl  --dataset_type musique --model_path /mnt/n0/models/llama3_8B_instruct --batch_size 1 --num_sample 1


# nohup python 0.triples_gen.py --dataset_path /mnt/n0/datasets/MuSiQue/musique_ans_v1.0_train.jsonl  --dataset_type musique --model_path /mnt/n0/models/llama3_8B_instruct --batch_size 20 --num_sample 6000 > triple.log 2>&1 &

# nohup python 0.triples_gen.py --dataset_path /mnt/n0/datasets/MuSiQue/musique_ans_v1.0_train.jsonl  --dataset_type musique --model_path /mnt/n0/models/llama3_8B_instruct --batch_size 20 --start_from 6000 --save_every 1000 > triple_v2.log 2>&1 &