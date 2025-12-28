# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.softmax_plus_one_transformer.configuration_softmax_plus_one_transformer import (
    SoftmaxPlusOneTransformerConfig,
)
from fla.models.softmax_plus_one_transformer.modeling_softmax_plus_one_transformer import (
    SoftmaxPlusOneTransformerForCausalLM,
    SoftmaxPlusOneTransformerModel,
)

AutoConfig.register(SoftmaxPlusOneTransformerConfig.model_type, SoftmaxPlusOneTransformerConfig)
AutoModel.register(SoftmaxPlusOneTransformerConfig, SoftmaxPlusOneTransformerModel)
AutoModelForCausalLM.register(SoftmaxPlusOneTransformerConfig, SoftmaxPlusOneTransformerForCausalLM)

__all__ = [
    "SoftmaxPlusOneTransformerConfig",
    "SoftmaxPlusOneTransformerForCausalLM",
    "SoftmaxPlusOneTransformerModel",
]
