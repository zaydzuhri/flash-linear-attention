# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.gpt_oss_sink_transformer.configuration_gpt_oss_sink_transformer import (
    GptOssSinkTransformerConfig,
)
from fla.models.gpt_oss_sink_transformer.modeling_gpt_oss_sink_transformer import (
    GptOssSinkTransformerForCausalLM,
    GptOssSinkTransformerModel,
)

AutoConfig.register(GptOssSinkTransformerConfig.model_type, GptOssSinkTransformerConfig)
AutoModel.register(GptOssSinkTransformerConfig, GptOssSinkTransformerModel)
AutoModelForCausalLM.register(GptOssSinkTransformerConfig, GptOssSinkTransformerForCausalLM)

__all__ = [
    "GptOssSinkTransformerConfig",
    "GptOssSinkTransformerForCausalLM",
    "GptOssSinkTransformerModel",
]
