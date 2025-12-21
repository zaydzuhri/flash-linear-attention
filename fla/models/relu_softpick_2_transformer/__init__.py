# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.relu_softpick_2_transformer.configuration_relu_softpick_2_transformer import (
    ReluSoftpick2TransformerConfig,
)
from fla.models.relu_softpick_2_transformer.modeling_relu_softpick_2_transformer import (
    ReluSoftpick2TransformerForCausalLM,
    ReluSoftpick2TransformerModel,
)

AutoConfig.register(ReluSoftpick2TransformerConfig.model_type, ReluSoftpick2TransformerConfig)
AutoModel.register(ReluSoftpick2TransformerConfig, ReluSoftpick2TransformerModel)
AutoModelForCausalLM.register(ReluSoftpick2TransformerConfig, ReluSoftpick2TransformerForCausalLM)

__all__ = [
    "ReluSoftpick2TransformerConfig",
    "ReluSoftpick2TransformerForCausalLM",
    "ReluSoftpick2TransformerModel",
]
