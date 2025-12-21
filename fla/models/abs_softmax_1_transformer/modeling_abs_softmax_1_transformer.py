# -*- coding: utf-8 -*-

from __future__ import annotations

from fla.models.abs_softmax_1_transformer.configuration_abs_softmax_1_transformer import (
    AbsSoftmax1TransformerConfig,
)
from fla.models.transformer.modeling_transformer import (
    TransformerForCausalLM,
    TransformerModel,
    TransformerPreTrainedModel,
)


class AbsSoftmax1TransformerPreTrainedModel(TransformerPreTrainedModel):
    config_class = AbsSoftmax1TransformerConfig


class AbsSoftmax1TransformerModel(TransformerModel):
    config_class = AbsSoftmax1TransformerConfig


class AbsSoftmax1TransformerForCausalLM(TransformerForCausalLM):
    config_class = AbsSoftmax1TransformerConfig
