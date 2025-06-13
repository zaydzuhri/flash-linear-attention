# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.transformer_mtp.configuration_transformer import MTPTransformerConfig
from fla.models.transformer_mtp.modeling_transformer import MTPTransformerForCausalLM, MTPTransformerModel

AutoConfig.register(MTPTransformerConfig.model_type, MTPTransformerConfig)
AutoModel.register(MTPTransformerConfig, MTPTransformerModel)
AutoModelForCausalLM.register(MTPTransformerConfig, MTPTransformerForCausalLM)


__all__ = ['TransformerConfig', 'TransformerForCausalLM', 'TransformerModel']
