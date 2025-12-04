from transformers import PretrainedConfig


class KBLaMConfig(PretrainedConfig):
    def __init__(
        self,
        base_model_name_or_path: str = "",
        kb_layer_frequency: int = 3,
        kb_scale_factor: float | None = None,
        top_k_kb: int = 100,
        dynamic_sparsify: bool = False,
        sep_query_head: bool = False,
        attn_implementation: str = "eager",
        format_short: bool = False,
        path_attn: bool = False,
        **kwargs,
    ):
        self.base_model_name_or_path = base_model_name_or_path
        self.kb_layer_frequency = kb_layer_frequency
        self.kb_scale_factor = kb_scale_factor
        self.top_k_kb = top_k_kb
        self.dynamic_sparsify = dynamic_sparsify
        self.sep_query_head = sep_query_head
        self.attn_implementation = attn_implementation
        self.format_short = format_short
        self.path_attn = path_attn

        super().__init__(**kwargs)
