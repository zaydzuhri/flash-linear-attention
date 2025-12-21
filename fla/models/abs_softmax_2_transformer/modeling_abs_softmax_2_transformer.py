# -*- coding: utf-8 -*-

from __future__ import annotations

from fla.models.abs_softmax_2_transformer.configuration_abs_softmax_2_transformer import (
    AbsSoftmax2TransformerConfig,
)
from fla.models.transformer.modeling_transformer import (
    TransformerForCausalLM,
    TransformerModel,
    TransformerPreTrainedModel,
)


class AbsSoftmax2TransformerPreTrainedModel(TransformerPreTrainedModel):
    config_class = AbsSoftmax2TransformerConfig


class AbsSoftmax2TransformerModel(TransformerModel):
    config_class = AbsSoftmax2TransformerConfig


class AbsSoftmax2TransformerForCausalLM(TransformerForCausalLM):
    config_class = AbsSoftmax2TransformerConfig
