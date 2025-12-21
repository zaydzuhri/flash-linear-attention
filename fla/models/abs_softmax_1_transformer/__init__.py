# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.abs_softmax_1_transformer.configuration_abs_softmax_1_transformer import (
    AbsSoftmax1TransformerConfig,
)
from fla.models.abs_softmax_1_transformer.modeling_abs_softmax_1_transformer import (
    AbsSoftmax1TransformerForCausalLM,
    AbsSoftmax1TransformerModel,
)

AutoConfig.register(AbsSoftmax1TransformerConfig.model_type, AbsSoftmax1TransformerConfig)
AutoModel.register(AbsSoftmax1TransformerConfig, AbsSoftmax1TransformerModel)
AutoModelForCausalLM.register(AbsSoftmax1TransformerConfig, AbsSoftmax1TransformerForCausalLM)

__all__ = [
    "AbsSoftmax1TransformerConfig",
    "AbsSoftmax1TransformerForCausalLM",
    "AbsSoftmax1TransformerModel",
]
