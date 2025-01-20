# -*- coding: utf-8 -*-

from typing import Dict, Optional

from transformers.configuration_utils import PretrainedConfig


class SCANConfig(PretrainedConfig):

    model_type = 'scan'
    # keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        hidden_size: int = 2048,
        window_size: int = 512,
        gate_logit_normalizer: Optional[int] = 8,
        clamp_min: Optional[float] = None,
        clamp_max: Optional[float] = None,
        hidden_ratio: Optional[int] = 4,
        intermediate_size: Optional[int] = None,
        num_hidden_layers: int = 24,
        num_heads: int = 4,
        num_kv_heads: Optional[int] = None,
        state_size: Optional[int] = 64,
        expand_k: float = 1,
        expand_v: float = 1,
        gate_act: str = 'softmax',
        use_output_gate: bool = False,
        use_norm: bool = True,
        hidden_act: str = "swish",
        elementwise_affine: Optional[bool] = True,
        max_position_embeddings: Optional[int] = 2048,
        norm_first: bool = True,
        norm_eps: float = 1e-6,
        attn: Optional[Dict] = None,
        use_cache: bool = True,
        pad_token_id: int = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = False,
        fuse_norm: bool = True,
        fuse_cross_entropy: bool = True,
        cut_cross_entropy: bool = True,
        vocab_size: int = 32000,
        **kwargs
    ):
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.gate_logit_normalizer = gate_logit_normalizer
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.state_size = state_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.gate_act = gate_act
        self.use_output_gate = use_output_gate
        self.use_norm = use_norm
        self.hidden_act = hidden_act
        self.elementwise_affine = elementwise_affine
        self.max_position_embeddings = max_position_embeddings
        self.norm_first = norm_first
        self.norm_eps = norm_eps
        self.attn = attn
        self.use_cache = use_cache
        self.initializer_range = initializer_range
        self.fuse_cross_entropy = fuse_cross_entropy
        self.cut_cross_entropy = cut_cross_entropy
        self.fuse_norm = fuse_norm
        self.vocab_size = vocab_size

        if attn is not None:
            if not isinstance(attn, Dict):
                raise ValueError("attn must be a dictionary")
            if 'layers' not in attn:
                raise ValueError("Layer indices must be provided to initialize hybrid attention layers")
            if 'num_heads' not in attn:
                raise ValueError("Number of heads must be provided to initialize hybrid attention layers")
            attn['num_kv_heads'] = attn.get('num_kv_heads', attn['num_heads'])
            attn['window_size'] = attn.get('window_size', None)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
