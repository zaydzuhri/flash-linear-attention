# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.mono_transformer.configuration_transformer import MonoTransformerConfig
from fla.models.mono_transformer.modeling_transformer import MonoTransformerForCausalLM, MonoTransformerModel

AutoConfig.register(MonoTransformerConfig.model_type, MonoTransformerConfig)
AutoModel.register(MonoTransformerConfig, MonoTransformerModel)
AutoModelForCausalLM.register(MonoTransformerConfig, MonoTransformerForCausalLM)


__all__ = ['MonoTransformerConfig', 'MonoTransformerForCausalLM', 'MonoTransformerModel']
