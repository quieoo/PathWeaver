# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import asdict
import argparse
import contextlib
import os
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs

# First Party
from common import (
    destroy_lmcache_engine,
    setup_lmcache_environment,
)


@contextlib.contextmanager
def build_llm_with_lmcache(lmcache_connector: str, model: str):

    if "llama3_8B" in model:
        max_model_len = 8192
    else:
        max_model_len = 32648

    ktc = KVTransferConfig(
        kv_connector=lmcache_connector,
        kv_role="kv_both",
    )

    llm_args = EngineArgs(
        model=model,
        kv_transfer_config=ktc,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.7,
        enable_prefix_caching=False,
        enforce_eager=True,
        
    )

    llm = LLM(**asdict(llm_args))
    try:
        yield llm
    finally:
        destroy_lmcache_engine()


def print_output(
    llm: LLM,
    prompt: list[int],
    sampling_params: SamplingParams,
    req_str: str,
):
    start = time.perf_counter()
    outputs = llm.generate(
        prompts={"prompt_token_ids": prompt}, sampling_params=sampling_params
    )
    elapsed = time.perf_counter() - start
    print("-" * 50)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")
        first_token_time = getattr(output, "first_token_time", None)
        if first_token_time is not None:
            print(f"TTFT: {first_token_time - start:.3f} seconds")
        else:
            print("TTFT: unavailable")
    print(f"Generation took {elapsed:.2f} seconds, {req_str} request done.")
    print("-" * 50)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--blend-special-str",
        default="# #",
        help="Specify the special separators to separate chunks (default: '# #')",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
    )

    parser.add_argument(
        "--max-local-cpu-size",
        type=float,
        default=5.0,
        help="Maximum LMCache local CPU cache size in GB.",
    )

    parser.add_argument(
        "--enable-sparse",
        action="store_true",
    )

    return parser.parse_args()


def test_run_0(llm, tokenizer, token_factor):
    # blend, default testing
    warmup_prompt = tokenizer.encode("Nice to meet you" * 5*token_factor)[1:]
    sys_prompt = [1, 733, 16289, 28793] + tokenizer.encode(
        "You are a very helpful assistant. "
        "Please answer the question with instructions."
    )
    chunk1_prompt = tokenizer.encode("Hello, how are you?" * 5*token_factor)[1:]
    chunk2_prompt = tokenizer.encode("Hello, what's up?" * 5*token_factor)[1:]
    chunk3_prompt = tokenizer.encode("Hi, what are you up to?" * 5*token_factor)[1:]
    blend_special_str = tokenizer.encode(os.getenv("LMCACHE_BLEND_SPECIAL_STR"))[1:]

    first_prompt = (
        sys_prompt
        + blend_special_str
        + chunk1_prompt
        + blend_special_str
        + chunk2_prompt
        + blend_special_str
        + chunk3_prompt
        + blend_special_str
        + tokenizer.encode("Hello, my name is")[1:]
        + [733, 28748, 16289, 28793]
    )

    second_prompt = (
        sys_prompt
        + blend_special_str
        + chunk2_prompt
        + blend_special_str
        + chunk1_prompt
        + blend_special_str
        + chunk3_prompt
        + blend_special_str
        + tokenizer.encode("Hello, how are you?")[1:]
        + [733, 28748, 16289, 28793]
    )

    third_prompt = (
        sys_prompt
        + blend_special_str
        + chunk2_prompt
        + blend_special_str
        + chunk1_prompt
        + blend_special_str
        + chunk3_prompt
        + blend_special_str
        + tokenizer.encode("Hello, what's up?")[1:]
        + [733, 28748, 16289, 28793]
    )

    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=128)

    print_output(llm, warmup_prompt, sampling_params, "warmup")

    # Print the first output
    print_output(llm, first_prompt, sampling_params, "first")

    time.sleep(1)

    # print the second output
    print_output(
        llm, second_prompt, sampling_params, "second (warming up blend code path)"
    )

    time.sleep(1)

    # print the third output
    print_output(llm, third_prompt, sampling_params, "third")

def test_run_1(llm, tokenizer, token_factor):
    # 首先分别发送每个独立文档，然后发送一次融合文档之后的请求，观察最后一次的重用情况
    # with prepare: TTFT: 0.256 seconds
    # without prepare: TTFT: 0.951 seconds


    chunk1_prompt = tokenizer.encode("Hello, how are you?" * 5*token_factor)[1:]
    chunk2_prompt = tokenizer.encode("Hello, what's up?" * 5*token_factor)[1:]
    chunk3_prompt = tokenizer.encode("Hi, what are you up to?" * 5*token_factor)[1:]
    blend_special_str = tokenizer.encode(os.getenv("LMCACHE_BLEND_SPECIAL_STR"))[1:]

    reuse_prompt = (
        chunk1_prompt
        + blend_special_str
        + chunk2_prompt
        + blend_special_str
        + chunk3_prompt
    )

    prepare_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)
    reuse_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=128)

    # print("Enable warmup")
    # print_output(llm, chunk1_prompt, prepare_params, "prepare_1")
    # print_output(llm, chunk2_prompt, prepare_params, "prepare_2")
    # print_output(llm, chunk3_prompt, prepare_params, "prepare_3")

    print("Disable warmup")
    print_output(llm, reuse_prompt, reuse_params, "reuse")
    


def main():
    args = parse_args()

    if "llama3_8B" in args.model:
        token_factor = 10
    else:
        token_factor = 100

    lmcache_connector = "LMCacheConnectorV1"
    model = args.model

    setup_lmcache_environment(
        blend_special_str=args.blend_special_str,
        local_cpu_size_gb=args.max_local_cpu_size,
        enable_sparse=args.enable_sparse,
    )

    tokenizer = AutoTokenizer.from_pretrained(model)

    with build_llm_with_lmcache(lmcache_connector, model) as llm:
        # Define the shared prompt and specific prompts

        # test_run_0(llm, tokenizer, token_factor)
        test_run_1(llm, tokenizer, token_factor)
        


if __name__ == "__main__":
    main()
