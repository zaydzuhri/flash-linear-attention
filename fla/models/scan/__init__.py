# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.scan.configuration_scan import SCANConfig
from fla.models.scan.modeling_scan import SCANForCausalLM, SCANModel

AutoConfig.register(SCANConfig.model_type, SCANConfig)
AutoModel.register(SCANConfig, SCANModel)
AutoModelForCausalLM.register(SCANConfig, SCANForCausalLM)


__all__ = ['SCANConfig', 'SCANForCausalLM', 'SCANModel']
