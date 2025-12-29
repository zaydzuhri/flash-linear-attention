# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.transformer_with_pruning.configuration_transformer_with_pruning import TransformerWithPruningConfig
from fla.models.transformer_with_pruning.modeling_transformer_with_pruning import TransformerWithPruningForCausalLM, TransformerWithPruningModel

AutoConfig.register(TransformerWithPruningConfig.model_type, TransformerWithPruningConfig)
AutoModel.register(TransformerWithPruningConfig, TransformerWithPruningModel)
AutoModelForCausalLM.register(TransformerWithPruningConfig, TransformerWithPruningForCausalLM)


__all__ = ['TransformerWithPruningConfig', 'TransformerWithPruningForCausalLM', 'TransformerWithPruningModel']
