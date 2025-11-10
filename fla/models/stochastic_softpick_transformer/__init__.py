# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.stochastic_softpick_transformer.configuration_stochastic_softpick_transformer import StochasticSoftpickTransformerConfig 
from fla.models.stochastic_softpick_transformer.modeling_stochastic_softpick_transformer import StochasticSoftpickTransformerForCausalLM, StochasticSoftpickTransformerModel 

AutoConfig.register(StochasticSoftpickTransformerConfig.model_type, StochasticSoftpickTransformerConfig)
AutoModel.register(StochasticSoftpickTransformerConfig, StochasticSoftpickTransformerModel)
AutoModelForCausalLM.register(StochasticSoftpickTransformerConfig, StochasticSoftpickTransformerForCausalLM)


__all__ = [
    'StochasticSoftpickTransformerConfig',
    'StochasticSoftpickTransformerForCausalLM', 
    'StochasticSoftpickTransformerModel'
]
