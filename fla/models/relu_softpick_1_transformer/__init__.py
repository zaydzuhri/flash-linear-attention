# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.relu_softpick_1_transformer.configuration_relu_softpick_1_transformer import (
    ReluSoftpick1TransformerConfig,
)
from fla.models.relu_softpick_1_transformer.modeling_relu_softpick_1_transformer import (
    ReluSoftpick1TransformerForCausalLM,
    ReluSoftpick1TransformerModel,
)

AutoConfig.register(ReluSoftpick1TransformerConfig.model_type, ReluSoftpick1TransformerConfig)
AutoModel.register(ReluSoftpick1TransformerConfig, ReluSoftpick1TransformerModel)
AutoModelForCausalLM.register(ReluSoftpick1TransformerConfig, ReluSoftpick1TransformerForCausalLM)

__all__ = [
    "ReluSoftpick1TransformerConfig",
    "ReluSoftpick1TransformerForCausalLM",
    "ReluSoftpick1TransformerModel",
]
