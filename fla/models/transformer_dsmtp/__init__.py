# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.transformer_dsmtp.configuration_transformer import DSMTPTransformerConfig
from fla.models.transformer_dsmtp.modeling_transformer import DSMTPTransformerForCausalLM, DSMTPTransformerModel

AutoConfig.register(DSMTPTransformerConfig.model_type, DSMTPTransformerConfig)
AutoModel.register(DSMTPTransformerConfig, DSMTPTransformerModel)
AutoModelForCausalLM.register(DSMTPTransformerConfig, DSMTPTransformerForCausalLM)


__all__ = ['DSMTPTransformerConfig', 'DSMTPTransformerForCausalLM', 'DSMTPTransformerModel']
