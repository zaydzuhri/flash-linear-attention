# -*- coding: utf-8 -*-

from __future__ import annotations

from fla.models.relu_softpick_2_transformer.configuration_relu_softpick_2_transformer import (
    ReluSoftpick2TransformerConfig,
)
from fla.models.transformer.modeling_transformer import (
    TransformerForCausalLM,
    TransformerModel,
    TransformerPreTrainedModel,
)


class ReluSoftpick2TransformerPreTrainedModel(TransformerPreTrainedModel):
    config_class = ReluSoftpick2TransformerConfig


class ReluSoftpick2TransformerModel(TransformerModel):
    config_class = ReluSoftpick2TransformerConfig


class ReluSoftpick2TransformerForCausalLM(TransformerForCausalLM):
    config_class = ReluSoftpick2TransformerConfig
