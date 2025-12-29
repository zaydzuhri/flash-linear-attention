# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.gated_attention_transformer.configuration_gated_attention_transformer import (
    GatedAttentionTransformerConfig,
)
from fla.models.gated_attention_transformer.modeling_gated_attention_transformer import (
    GatedAttentionTransformerForCausalLM,
    GatedAttentionTransformerModel,
)

AutoConfig.register(GatedAttentionTransformerConfig.model_type, GatedAttentionTransformerConfig)
AutoModel.register(GatedAttentionTransformerConfig, GatedAttentionTransformerModel)
AutoModelForCausalLM.register(GatedAttentionTransformerConfig, GatedAttentionTransformerForCausalLM)

__all__ = [
    "GatedAttentionTransformerConfig",
    "GatedAttentionTransformerForCausalLM",
    "GatedAttentionTransformerModel",
]
