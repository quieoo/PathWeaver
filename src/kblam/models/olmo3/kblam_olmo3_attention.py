import torch
from transformers.models.olmo3.modeling_olmo3 import Olmo3Attention

# 你 Phase 2 的计数逻辑可以保留，不影响

def _kb_gate_from_inputs(kb_kvs=None, kb_adj=None, kb_config=None, device=None, dtype=None) -> torch.Tensor:
    """
    Phase 3.1:
    - 从 kb 输入构造一个标量 gate（Tensor 标量）
    - 目标：便宜、稳定、可预测
    """
    if kb_config is None:
        return torch.tensor(0.0, device=device, dtype=dtype)

    scale = float(getattr(kb_config, "kb_gate_scale", 0.0))
    if scale == 0.0:
        return torch.tensor(0.0, device=device, dtype=dtype)

    # 允许没有 kb 输入时也可跑（gate=0）
    if kb_kvs is None and kb_adj is None:
        return torch.tensor(0.0, device=device, dtype=dtype)

    g = 0.0

    # 1) kb_kvs: (keys, values)
    if kb_kvs is not None:
        try:
            kb_keys, kb_values = kb_kvs
            if kb_keys is not None:
                g = g + kb_keys.float().mean()
            elif kb_values is not None:
                g = g + kb_values.float().mean()
        except Exception:
            # 如果格式不符，保持 gate=0（不炸）
            pass

    # 2) kb_adj: 稀疏/稠密都行，取一个均值（可解释：平均边权）
    if kb_adj is not None:
        try:
            if getattr(kb_adj, "is_sparse", False):
                # sparse tensor
                g = g + kb_adj.values().float().mean()
            else:
                g = g + kb_adj.float().mean()
        except Exception:
            pass

    # 缩放 + clamp，避免极端值
    gate = g * scale
    gate = torch.clamp(gate, float(getattr(kb_config, "kb_gate_clip", 1e-2)) * -1,
                      float(getattr(kb_config, "kb_gate_clip", 1e-2)))

    return gate.to(device=device, dtype=dtype)


class KBLAMOlmo3Attention(Olmo3Attention):
    def forward(self, *args, kb_kvs=None, kb_adj=None, kb_config=None, **kwargs):
        # 仍然完全依赖官方实现
        attn_output, *rest = super().forward(*args, **kwargs)

        # Phase 3.1: KB 驱动 gate 注入（最小侵入点）
        gate = _kb_gate_from_inputs(
            kb_kvs=kb_kvs,
            kb_adj=kb_adj,
            kb_config=kb_config,
            device=attn_output.device,
            dtype=attn_output.dtype,
        )
        if gate != 0:  # gate 是标量 tensor，这里是 python 分支；想更纯可去掉分支
            attn_output = attn_output + gate

        return (attn_output, *rest)
