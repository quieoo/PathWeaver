import math
import torch
from .kblam_path import apply_kblam_path_attention

PADDING_VALUE = torch.finfo(torch.bfloat16).min

def apply_kblam_sep_query_head(
    *,
    attn_weights,
    query_states_2,
    kb_keys,
    kb_len: int,
    kb_config,
    layer_idx: int,
    head_dim: int,
):
    """
    等价于 llama3_model.py 中 sep_query_head 分支
    """
    if (
        not kb_config.sep_query_head
        or kb_len == 0
        or layer_idx % kb_config.kb_layer_frequency != 0
    ):
        return attn_weights

    # 重新计算 KB logits（用 query_states_2）
    attn_weights_2 = torch.matmul(
        query_states_2, kb_keys.transpose(2, 3)
    ) / math.sqrt(head_dim)

    if kb_config.kb_scale_factor is not None:
        attn_weights_2 = attn_weights_2 * kb_config.kb_scale_factor

    # 丢弃原来 text-query 对 KB 的 logits
    attn_weights_text = attn_weights[..., kb_len:]

    return torch.cat([attn_weights_2, attn_weights_text], dim=-1)

    

def apply_kblam_attention(
    *,
    query_states,
    query_states_2,
    key_states,
    value_states,
    attention_mask,
    kb_kvs,
    kb_config,
    kb_adj,
    layer_idx: int,
    num_heads: int,
    head_dim: int,
    num_hidden_layers: int,
):
    """
    从 llama3_model.py 抽出的 KB 注入主逻辑
    返回：
        key_states, value_states, attention_mask, kb_len
    """
    if kb_kvs is None or kb_config is None:
        return key_states, value_states, attention_mask, 0

    if layer_idx % kb_config.kb_layer_frequency != 0:
        return key_states, value_states, attention_mask, 0

    kb_keys, kb_values = kb_kvs
    bsz = query_states.size(0)

    # ---- reshape / slice（原逻辑 1:1）----
    if kb_keys.dim() == 2:
        kb_len = kb_keys.shape[0]
        kb_idx = layer_idx // kb_config.kb_layer_frequency
        kb_keys = kb_keys.reshape(
            kb_len,
            1 + num_hidden_layers // kb_config.kb_layer_frequency,
            -1,
        )[:, kb_idx]
        kb_values = kb_values.reshape(
            kb_len,
            1 + num_hidden_layers // kb_config.kb_layer_frequency,
            -1,
        )[:, kb_idx]
        kb_keys = kb_keys.view(kb_len, num_heads, head_dim).transpose(0, 1)
        kb_values = kb_values.view(kb_len, num_heads, head_dim).transpose(0, 1)
        kb_keys = kb_keys.unsqueeze(0).expand(bsz, num_heads, kb_len, head_dim)
        kb_values = kb_values.unsqueeze(0).expand(bsz, num_heads, kb_len, head_dim)
    else:
        kb_len = kb_keys.shape[1]
        kb_idx = layer_idx // kb_config.kb_layer_frequency
        kb_keys = kb_keys.view(
            bsz,
            kb_len,
            1 + num_hidden_layers // kb_config.kb_layer_frequency,
            -1,
        )[:, :, kb_idx]
        kb_values = kb_values.view(
            bsz,
            kb_len,
            1 + num_hidden_layers // kb_config.kb_layer_frequency,
            -1,
        )[:, :, kb_idx]
        kb_keys = kb_keys.view(bsz, kb_len, num_heads, head_dim).transpose(1, 2)
        kb_values = kb_values.view(bsz, kb_len, num_heads, head_dim).transpose(1, 2)

    key_states = torch.cat([kb_keys, key_states], dim=2)
    value_states = torch.cat([kb_values, value_states], dim=2)

    kb_mask = attention_mask.new_zeros(bsz, 1, attention_mask.size(2), kb_len)
    padding_mask = torch.all(attention_mask < 0, -1, keepdim=True)
    kb_mask = padding_mask * PADDING_VALUE + (~padding_mask) * kb_mask
    attention_mask = torch.cat([kb_mask, attention_mask], dim=-1)

    return key_states, value_states, attention_mask, kb_len
