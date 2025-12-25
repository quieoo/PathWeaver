import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import evaluate
import nltk
import numpy as np
import torch
import transformers
from tqdm import tqdm
from transformers import AutoTokenizer, logging

from kblam.kb_encoder import KBEncoder
from kblam.models.kblam_config import KBLaMConfig
from kblam.models.llama3_model import KblamLlamaForCausalLM
from kblam.models.phi3_model import KBLaMPhi3ForCausalLM
from kblam.utils.data_utils import augment_row, generate_multi_entity_qa
from kblam.utils.eval_utils import (
    instruction_prompts,
    instruction_prompts_multi_entities,
    zero_shot_prompt,
    zero_shot_prompt_multi_entities,
    _format_Q_llama,
    _format_Q_phi3,
    model_prune_format_mapping,
    answer_question,
    softmax,
)
from kblam.utils.train_utils import get_kb_embd
# 增加
from sentence_transformers import SentenceTransformer 
from typing import List, Tuple
# kmeans++  
from kblam.utils.kmeans import KMeansPlusPlus
import torch.nn.functional as F

logging.set_verbosity_warning()

rouge = evaluate.load("rouge")
bert_score = evaluate.load("bertscore")

# 参数
parser = argparse.ArgumentParser(description="Evaluation script")

parser.add_argument(
    "--dataset_dir", type=str, help="Directory containing the dataset"
)
parser.add_argument(
    "--encoder_dir", type=str, help="Directory containing the encoder model"
)
parser.add_argument(
    "--encoder_spec",
    type=str,
    default="OAI",
    help="Specification for the encoder model",
)
parser.add_argument(
    "--fancy_instruction",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Whether to use fancy instructions",
)
parser.add_argument(
    "--kb_layer_frequency",
    type=int,
    default=3,
    help="Frequency of knowledge base layers",
)
parser.add_argument(
    "--kb_scale_factor",
    type=int,
    default=None,
    help="Scaling factor for knowledge base",
)
parser.add_argument(
    "--kb_size", type=int, default=200, help="Size of the knowledge base"
)
parser.add_argument(
    "--llm_base_dir",
    type=str,
    help="llm to load, can be HF location or local directory",
)
parser.add_argument(
    "--llm_type",
    type=str,
    default="phi3",
    choices=["llama3", "phi3"],
    help="Type of language model to use",
)
parser.add_argument(
    "--model_dir", type=str, help="Directory containing the model"
)
parser.add_argument("--save_dir", type=str, help="Directory to save outputs")
parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
parser.add_argument(
    "--test_dataset", type=str, help="Source of test KB (assumes KV pair format)"
)
parser.add_argument(
    "--precomputed_embed_keys_path", type=str, help="Path to precomputed key embeddings"
)
parser.add_argument(
    "--precomputed_embed_values_path",
    type=str,
    help="Path to precomputed value embeddings",
)
parser.add_argument(
    "--query_head_path", type=str, default="", help="Path to load KB head from"
)

class KBRetriever:
    def __init__(
        self,
        encoder: KBEncoder,
        dataset: List[Dict],
        precomputed_embed_keys_path: Optional[str] = None,
        precomputed_embed_values_path: Optional[np.ndarray] = None,
    ):
        self.encoder = encoder
        self.dataset = dataset
        if precomputed_embed_keys_path is not None:
            self.key_embds = np.load(precomputed_embed_keys_path).astype("float32")
        else:
            self.key_embds = None
        if precomputed_embed_values_path is not None:
            self.value_embds = np.load(precomputed_embed_values_path).astype("float32")
        else:
            self.value_embds = None

        if precomputed_embed_keys_path is not None:
            assert len(dataset) == len(self.key_embds)

    def _use_cached_embd(self):
        if self.key_embds is not None and self.value_embds is not None:
            return True
        else:
            return False

    def get_key_embeddings(self, batch_indices):
        if self._use_cached_embd():
            return get_kb_embd(
                self.encoder,
                batch_indices,
                precomputed_embd=(self.key_embds, self.value_embds),
            )
        else:
            return get_kb_embd(self.encoder, batch_indices, kb_dict=self.dataset)
        
def _prepare_models(
    encoder_spec,
    encoder_path,
    llm_type,
    llm_base_dir,
    model_path,
    query_head_path,
    kb_layer_frequency,
    kb_scale_factor,
):
    tokenizer = AutoTokenizer.from_pretrained(
        llm_base_dir, trust_remote_code=True, padding_side="left"
    )
    tokenizer.pad_token = "^"

    if llm_type == "llama3":
        if query_head_path:
            model = KblamLlamaForCausalLM.from_pretrained(
                model_path,
                device_map="cuda",
                torch_dtype="auto",
                trust_remote_code=True,
            )
            model.load_query_head(query_head_path)
        else:
            model = KblamLlamaForCausalLM.from_pretrained(
                model_path,
                device_map="cuda",
                torch_dtype="auto",
                trust_remote_code=True,
            )
    else:
        model = KBLaMPhi3ForCausalLM.from_pretrained(
            model_path,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
        )
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.eval()

    # config = model.config.to_dict()
    kb_config = KBLaMConfig(
        #sep_query_head=False,
        sep_query_head=True,
        kb_layer_frequency=kb_layer_frequency,
        kb_scale_factor=kb_scale_factor,
        # 打开动态稀疏性
        dynamic_sparsify=True,
        top_k_kb=1,
    )
    print(kb_config)
    # config.update(kb_config.to_dict())
    # new_config = KBLaMConfig(**config)
    # model.config = new_config

    encoder = KBEncoder(
        #encoder_name=encoder_spec.upper(),
        encoder_name=encoder_spec,
        projector_type="linear",
        endpoint_url="",
        out_dim=model.config.hidden_size
        * (model.config.num_hidden_layers // kb_layer_frequency + 1),
        frozen_base_model=True,
        projector_kwargs={"mlp_depth": 1, "mlp_hidden_dim": 512},
        device=torch.device("cuda"),
    )

    encoder.load_state_dict(torch.load(encoder_path))
    return tokenizer, encoder, model, kb_config

def eval_generate():
    """Evaluate generation using KB"""
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    encoder_model_spec = args.encoder_spec
    encoder_path = args.encoder_dir
    kb_layer_frequency = args.kb_layer_frequency
    kb_scale_factor = args.kb_scale_factor
    kb_size = args.kb_size
    llm_base_dir = args.llm_base_dir
    llm_type = args.llm_type
    model_path = args.model_dir
    seed = args.seed
    test_dataset = args.test_dataset
    query_head_path = args.query_head_path
    precomputed_embed_keys_path = args.precomputed_embed_keys_path
    precomputed_embed_values_path = args.precomputed_embed_values_path

    dataset = json.load(open(os.path.join(dataset_dir, test_dataset)))

    tokenizer, encoder, model, kb_config = _prepare_models(
        encoder_model_spec,
        encoder_path,
        llm_type,
        llm_base_dir,
        model_path,
        query_head_path,
        kb_layer_frequency,
        kb_scale_factor,
    )

    kb_retriever = KBRetriever(
        encoder,
        dataset,
        precomputed_embed_keys_path=precomputed_embed_keys_path,
        precomputed_embed_values_path=precomputed_embed_values_path,
    )

    # perform_eval

    # ----------- 取出 embedding 并转成 torch.float32 -----------
    all_indices = np.arange(len(kb_retriever.dataset))
    print("total dataset size:", len(kb_retriever.dataset))
    kb_key_emb = torch.tensor(kb_retriever.key_embds[all_indices], dtype=torch.float32, device="cuda")

    # ----------- 做 kmeans++ 聚类 -----------
    codebook_size = 1024  
    kmeans = KMeansPlusPlus(n_clusters=codebook_size, device="cuda")
    kmeans.fit(kb_key_emb)
    codebook = kmeans.centroids          # [codebook_size, dim]
    vq_ids = kmeans.vq_ids               # [num_kb_tokens]
    print("Codebook shape:", codebook.shape)
    print("vq_ids shape:", vq_ids.shape) # torch.Size([num_kb_tokens])

    # 保存结果
    torch.save(codebook, "./codebook/llama-3-8b-synthetic-embd-codebook.pt")
    np.save("./codebook/llama-3-8b-synthetic-embd-vqids.npy", vq_ids.cpu().numpy())


if __name__ == "__main__":
    eval_generate()
