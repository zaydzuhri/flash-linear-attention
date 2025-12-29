# -*- coding: utf-8 -*-

from __future__ import annotations

from fla.models.softmax_plus_one_transformer.configuration_softmax_plus_one_transformer import (
    SoftmaxPlusOneTransformerConfig,
)
from fla.models.transformer.modeling_transformer import (
    TransformerForCausalLM,
    TransformerModel,
    TransformerPreTrainedModel,
)


class SoftmaxPlusOneTransformerPreTrainedModel(TransformerPreTrainedModel):
    config_class = SoftmaxPlusOneTransformerConfig


class SoftmaxPlusOneTransformerModel(TransformerModel):
    config_class = SoftmaxPlusOneTransformerConfig


class SoftmaxPlusOneTransformerForCausalLM(TransformerForCausalLM):
    config_class = SoftmaxPlusOneTransformerConfig
