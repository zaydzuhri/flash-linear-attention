# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.abs_softmax_2_transformer.configuration_abs_softmax_2_transformer import (
    AbsSoftmax2TransformerConfig,
)
from fla.models.abs_softmax_2_transformer.modeling_abs_softmax_2_transformer import (
    AbsSoftmax2TransformerForCausalLM,
    AbsSoftmax2TransformerModel,
)

AutoConfig.register(AbsSoftmax2TransformerConfig.model_type, AbsSoftmax2TransformerConfig)
AutoModel.register(AbsSoftmax2TransformerConfig, AbsSoftmax2TransformerModel)
AutoModelForCausalLM.register(AbsSoftmax2TransformerConfig, AbsSoftmax2TransformerForCausalLM)

__all__ = [
    "AbsSoftmax2TransformerConfig",
    "AbsSoftmax2TransformerForCausalLM",
    "AbsSoftmax2TransformerModel",
]
