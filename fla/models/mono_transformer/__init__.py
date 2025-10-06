# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.mono_transformer.configuration_transformer import MonoTransformerConfig
from fla.models.mono_transformer.modeling_transformer import MonoTransformerForCausalLM, MonoTransformerModel

AutoConfig.register(MonoTransformerConfig.model_type, MonoTransformerConfig, exist_ok=True)
AutoModel.register(MonoTransformerConfig, MonoTransformerModel, exist_ok=True)
AutoModelForCausalLM.register(MonoTransformerConfig, MonoTransformerForCausalLM, exist_ok=True)


__all__ = ['MonoTransformerConfig', 'MonoTransformerForCausalLM', 'MonoTransformerModel']
