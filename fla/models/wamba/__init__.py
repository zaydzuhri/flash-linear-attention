# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.wamba.configuration_wamba import WambaConfig
from fla.models.wamba.modeling_wamba import (WambaBlock, WambaForCausalLM,
                                             WambaModel)

AutoConfig.register(WambaConfig.model_type, WambaConfig, True)
AutoModel.register(WambaConfig, WambaModel, True)
AutoModelForCausalLM.register(WambaConfig, WambaForCausalLM, True)


__all__ = ['WambaConfig', 'WambaForCausalLM', 'WambaModel', 'WambaBlock']
