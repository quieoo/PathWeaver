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
        current_step: int = 1,
        total_steps: int = 1,
        base_embeder_path: str | None = None,
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
        self.current_step = current_step
        self.total_steps = total_steps
        self.base_embeder_path = base_embeder_path

        self.debug_bias = 0.0

        super().__init__(**kwargs)
