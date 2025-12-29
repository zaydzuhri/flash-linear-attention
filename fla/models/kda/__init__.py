
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.kda.configuration_kda import KDAConfig
from fla.models.kda.modeling_kda import KDAForCausalLM, KDAModel

AutoConfig.register(KDAConfig.model_type, KDAConfig)
AutoModel.register(KDAConfig, KDAModel)
AutoModelForCausalLM.register(KDAConfig, KDAForCausalLM)

__all__ = ['KDAConfig', 'KDAForCausalLM', 'KDAModel']
