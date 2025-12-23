# -*- coding: utf-8 -*-

from __future__ import annotations

from fla.models.gated_attention_transformer.configuration_gated_attention_transformer import (
    GatedAttentionTransformerConfig,
)
from fla.models.transformer.modeling_transformer import (
    TransformerForCausalLM,
    TransformerModel,
    TransformerPreTrainedModel,
)


class GatedAttentionTransformerPreTrainedModel(TransformerPreTrainedModel):
    config_class = GatedAttentionTransformerConfig


class GatedAttentionTransformerModel(TransformerModel):
    config_class = GatedAttentionTransformerConfig


class GatedAttentionTransformerForCausalLM(TransformerForCausalLM):
    config_class = GatedAttentionTransformerConfig
