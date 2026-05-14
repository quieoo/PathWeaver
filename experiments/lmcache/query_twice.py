import os
import time
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "blending.yaml"
os.environ["LMCACHE_CONFIG_FILE"] = str(DEFAULT_CONFIG_PATH)

MODEL = "/mnt/n0/models/qwen3-4B/"
MAX_CHARS_PER_FILE = 60000
MAX_TOKENS = 4096
TEMPERATURE = 0.0

SYSTEM_PROMPT = "You are a helpful assistant."
QUESTION_1 = "Summarize the two documents and explain one important difference."
QUESTION_2 = "Now answer the same question for the reordered documents."


def read_text(path: str, max_chars: int | None = None) -> str:
    text = Path(path).read_text(encoding="utf-8")
    if max_chars is not None:
        text = text[:max_chars]
    return text


tokenizer = AutoTokenizer.from_pretrained(MODEL)
sep_ids = tokenizer.encode(" # # ", add_special_tokens=False)


def enc(text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def build_prompt_token_ids(file_a: str, file_b: str, question: str) -> list[int]:
    return enc(SYSTEM_PROMPT) + sep_ids + enc(file_a) + sep_ids + enc(file_b) + sep_ids + enc(question)


def install_lmcache_blending_startup_patch() -> str:
    from lmcache.integration.vllm.utils import ENGINE_NAME
    from lmcache.v1.compute.models.utils import VLLMModelTracker
    from vllm.v1.worker.gpu_worker import GPUWorker

    original_initialize_from_config = GPUWorker.initialize_from_config
    if getattr(original_initialize_from_config, "_pathweaver_lmcache_patch", False):
        return "LMCache blending startup patch already installed."

    def patched_initialize_from_config(self: Any, kv_cache_config: Any) -> None:
        # LMCache 0.4.2 expects the vLLM model to be registered before
        # connector initialization, but vLLM 0.18.0 never does it.
        try:
            VLLMModelTracker.register_model(ENGINE_NAME, self.model_runner.get_model())
        except Exception as exc:
            print(f"Warning: failed to pre-register vLLM model for blending: {exc}")
        return original_initialize_from_config(self, kv_cache_config)

    patched_initialize_from_config._pathweaver_lmcache_patch = True
    GPUWorker.initialize_from_config = patched_initialize_from_config
    return "Installed runtime patch for LMCache blending model registration."


def run_once(llm: LLM, sampling_params: SamplingParams, tag: str, prompt_ids: list[int]):
    print(f"\n=== {tag} ===")
    print(f"prompt tokens  = {len(prompt_ids)}")
    print(f"prompt preview = {prompt_ids[:30]}")

    t0 = time.perf_counter()
    outputs = llm.generate(prompts={"prompt_token_ids": prompt_ids}, sampling_params=sampling_params)
    t1 = time.perf_counter()

    answer = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
    print(f"E2E           = {t1 - t0:.3f}s")
    print(f"answer        = {answer!r}")
    print("-" * 100)


def main():
    file1 = read_text(SCRIPT_DIR / "test_file_1.txt", MAX_CHARS_PER_FILE)
    file2 = read_text(SCRIPT_DIR / "test_file_2.txt", MAX_CHARS_PER_FILE)

    print(f"Loaded test_file_1.txt chars = {len(file1)}")
    print(f"Loaded test_file_2.txt chars = {len(file2)}")
    print(install_lmcache_blending_startup_patch())
    print(f"Using LMCache config = {os.environ['LMCACHE_CONFIG_FILE']}")

    prompt1 = build_prompt_token_ids(file1, file2, QUESTION_1)
    prompt2 = build_prompt_token_ids(file2, file1, QUESTION_2)

    llm = LLM(
        model=MODEL,
        kv_transfer_config={"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"},
        enable_prefix_caching=False,
        enforce_eager=True,
        max_model_len=16384,
    )
    sp = SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_TOKENS)

    # 第一次：populate cache
    run_once(llm, sp, "round1_file1_then_file2", prompt1)

    # 第二次：reordered, should trigger blending reuse
    run_once(llm, sp, "round2_file2_then_file1", prompt2)


if __name__ == "__main__":
    main()
