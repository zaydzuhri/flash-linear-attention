# -*- coding: utf-8 -*-

from __future__ import annotations

from fla.models.relu_softpick_1_transformer.configuration_relu_softpick_1_transformer import (
    ReluSoftpick1TransformerConfig,
)
from fla.models.transformer.modeling_transformer import (
    TransformerForCausalLM,
    TransformerModel,
    TransformerPreTrainedModel,
)


class ReluSoftpick1TransformerPreTrainedModel(TransformerPreTrainedModel):
    config_class = ReluSoftpick1TransformerConfig


class ReluSoftpick1TransformerModel(TransformerModel):
    config_class = ReluSoftpick1TransformerConfig


class ReluSoftpick1TransformerForCausalLM(TransformerForCausalLM):
    config_class = ReluSoftpick1TransformerConfig
