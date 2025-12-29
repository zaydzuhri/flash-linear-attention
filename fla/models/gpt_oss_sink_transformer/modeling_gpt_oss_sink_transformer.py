# -*- coding: utf-8 -*-

from __future__ import annotations

from fla.models.gpt_oss_sink_transformer.configuration_gpt_oss_sink_transformer import (
    GptOssSinkTransformerConfig,
)
from fla.models.transformer.modeling_transformer import (
    TransformerForCausalLM,
    TransformerModel,
    TransformerPreTrainedModel,
)


class GptOssSinkTransformerPreTrainedModel(TransformerPreTrainedModel):
    config_class = GptOssSinkTransformerConfig


class GptOssSinkTransformerModel(TransformerModel):
    config_class = GptOssSinkTransformerConfig


class GptOssSinkTransformerForCausalLM(TransformerForCausalLM):
    config_class = GptOssSinkTransformerConfig
