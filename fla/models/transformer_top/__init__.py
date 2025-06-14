# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.transformer_top.configuration_transformer import TOPTransformerConfig
from fla.models.transformer_top.modeling_transformer import TOPTransformerForCausalLM, TOPTransformerModel

AutoConfig.register(TOPTransformerConfig.model_type, TOPTransformerConfig)
AutoModel.register(TOPTransformerConfig, TOPTransformerModel)
AutoModelForCausalLM.register(TOPTransformerConfig, TOPTransformerForCausalLM)


__all__ = ['TOPTransformerConfig', 'TOPTransformerForCausalLM', 'TOPTransformerModel']
