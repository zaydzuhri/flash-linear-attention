# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from dataclasses import dataclass
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

import triton
import triton.language as tl

from fla.layers.attn import Attention
from fla.models.transformer_mtp.configuration_transformer import MTPTransformerConfig
from fla.models.utils import Cache
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss
from fla.modules import GatedMLP as TransformerMLP
from fla.modules import RMSNorm

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack


logger = logging.get_logger(__name__)

class SequentialHeadsCustomBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, trunk_output, lm_head, norm_layer, logits_to_keep, *prediction_heads):
        # We now need the norm layer in the forward pass calculation
        ctx.prediction_heads = prediction_heads
        ctx.lm_head = lm_head
        ctx.norm_layer = norm_layer 
        ctx.logits_to_keep = logits_to_keep
        ctx.save_for_backward(trunk_output)

        latents = []
        for head in prediction_heads:
            # Assuming head forward signature is `head(hidden_states)`
            latent = head(trunk_output)[0]
            latents.append(latent)

        latents_stacked = torch.stack(latents, dim=-2)
        # Apply the final norm before the lm_head
        normalized_latents = norm_layer(latents_stacked)
        all_logits = lm_head(normalized_latents[:, -logits_to_keep:])
        return all_logits

    @staticmethod
    def backward(ctx, grad_output):
        trunk_output, = ctx.saved_tensors
        prediction_heads = ctx.prediction_heads
        lm_head = ctx.lm_head
        norm_layer = ctx.norm_layer
        logits_to_keep = ctx.logits_to_keep

        d = trunk_output.detach().requires_grad_(True)
        grad_output_per_head = grad_output.unbind(dim=2)
        
        # We need to manually handle the backward pass for the final norm layer once
        # before the loop, as its gradient depends on all heads.
        # To do this, we reconstruct the input to the lm_head and do a backward pass.
        with torch.enable_grad():
            # Re-run the head computations to get the input to the norm layer
            latents = []
            for head in prediction_heads:
                latents.append(head(d)[0])
            latents_stacked = torch.stack(latents, dim=-2)
            latents_stacked.requires_grad_(True)
            # The part of the graph we need to backprop through first
            normalized_latents = norm_layer(latents_stacked)
        
        # Backpropagate through the lm_head and norm_layer
        normalized_latents.backward(lm_head.weight.grad @ grad_output)
        
        # Now, `latents_stacked.grad` contains the sum of gradients from all heads
        # just before the final normalization. We can now unbind it.
        grad_per_head_latent = latents_stacked.grad.unbind(dim=-2)

        # Now, backpropagate through each head individually.
        for i, head in enumerate(prediction_heads):
            with torch.enable_grad():
                head_latent = head(d)[0]
            # Backpropagate using the gradient for this specific head's output
            head_latent.backward(gradient=grad_per_head_latent[i])

        num_nones = 2 + len(prediction_heads) # for lm_head, norm_layer, and *prediction_heads
        return (d.grad,) + (None,) * num_nones

def seq_to_mtp(
    long_input_ids: torch.Tensor,
    model_seq_len: int,
    n_future_tokens: int
) -> torch.Tensor:
    """
    Generates a tensor of future targets on the fly from a long input sequence.

    This version assumes `long_input_ids` contains both the tokens for the model's
    input AND the future tokens needed for the labels.
    It extracts the correct targets without adding artificial padding.

    Args:
        long_input_ids (torch.Tensor): The input sequences from the dataloader,
                                       shape (B, T + n_future_tokens).
        model_seq_len (int): The sequence length `T` that the model processes.
        n_future_tokens (int): The number of future tokens to predict for each time step.

    Returns:
        torch.Tensor: The target tensor of shape (B, T, n_future_tokens).
                      y[b, t, k] corresponds to the (k+1)-th token after input_ids[b, t].
    """
    B, total_len = long_input_ids.shape
    assert total_len >= model_seq_len + n_future_tokens, \
        "long_input_ids must be at least model_seq_len + n_future_tokens long."

    # 1. Create sliding windows (views) over the long tensor.
    # .unfold() is a highly efficient way to create sliding windows.
    # We create windows of size `n_future_tokens + 1`. For each time step `t`,
    # the window will contain the input token and its `n_future_tokens` targets.
    # Example (n=3, window_size=4):
    # For t=0, window is [t0, t1, t2, t3]
    # For t=1, window is [t1, t2, t3, t4]
    # Shape of windows: (B, total_len - n_future_tokens, n_future_tokens + 1)
    windows = long_input_ids.unfold(dimension=1, size=n_future_tokens + 1, step=1)

    # 2. Slice the windows to get only the targets.
    # We slice off the first element of each window (the input token itself)
    # to keep only the future tokens.
    # Example window [t0, t1, t2, t3] -> becomes targets [t1, t2, t3]
    all_targets = windows[:, :, 1:]

    # 3. Trim the result to match the model's output sequence length.
    # We only need the targets for the first `model_seq_len` positions.
    output_targets = all_targets[:, :model_seq_len, :]

    return output_targets.transpose(1, 2)


@dataclass
class MTPLMOutputWithPast(CausalLMOutputWithPast):
    ntp_loss: Optional[torch.FloatTensor] = None
    mtp_loss: Optional[torch.FloatTensor] = None

class MTPTransformerBlock(nn.Module):

    def __init__(self, config: MTPTransformerConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.attn = Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            qkv_bias=config.qkv_bias,
            qk_norm=config.qk_norm,
            window_size=config.window_size,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            layer_idx=layer_idx
        )

        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.mlp = TransformerMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs: Unpack[Any]
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs
        )
        if self.config.fuse_norm:
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attentions,)

        if use_cache:
            outputs += (past_key_values,)

        return outputs


class MTPTransformerPreTrainedModel(PreTrainedModel):

    config_class = MTPTransformerConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['MTPTransformerBlock']
    _supports_cache_class = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(
        self,
        module: nn.Module,
        rescale_prenorm_residual: bool = False,
        num_residuals_per_layer: int = 2,
    ):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()

        if rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            p = None
            if hasattr(module, 'o_proj'):
                p = module.o_proj.weight
            elif hasattr(module, 'down_proj'):
                p = module.down_proj.weight
            if p is not None:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)


class MTPTransformerModel(MTPTransformerPreTrainedModel):

    def __init__(
        self,
        config: MTPTransformerConfig
    ) -> MTPTransformerModel:
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([MTPTransformerBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers - config.n_future_tokens)])
        self.extra_heads = nn.ModuleList([MTPTransformerBlock(config, layer_idx) for layer_idx in range(config.n_future_tokens)])
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)

        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_all_heads: bool = False, # if Training, this is True
        **kwargs: Unpack[Any]
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if output_attentions:
            warnings.warn(
                "`TransformerModel` does not support output attention weights now, so `output_attentions` is set to `False`."
            )
            output_attentions = False
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_custom_backward = self.config.use_custom_backward and self.training 
        if self.training and return_all_heads is False:
            logger.warning_once(
                "`return_all_heads=False` is incompatible with training. Setting `return_all_heads=True`..."
            )
            return_all_heads = True

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = Cache.from_legacy_cache(past_key_values)

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        # embed positions
        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        next_cache = None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    **kwargs
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    **kwargs
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_attns += (layer_outputs[1],)

        trunk = hidden_states

        n_heads_to_use = self.config.n_future_tokens if return_all_heads else 1
        prediction_heads_to_use = self.extra_heads[:n_heads_to_use]

        if use_custom_backward and self.training:
            # all_logits = SequentialHeadsCustomBackward.apply(trunk, self.lm_head, *prediction_heads)
            hidden_states = trunk # return hidden states and apply custom backward on the MTPTransformersLM
        else:
            latents = []
            for i, layer in enumerate(prediction_heads_to_use):
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        layer.__call__,
                        trunk, # Use trunk instead of hidden states
                        attention_mask,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        **kwargs
                    )
                else:
                    layer_outputs = layer(
                        trunk,  # Use trunk instead of hidden states
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        **kwargs
                    )
                hidden_states = layer_outputs[0]
                latents.append(hidden_states)

                if use_cache:
                    next_cache = layer_outputs[2 if output_attentions else 1]
                
                if output_attentions:
                    all_attns += (layer_outputs[1],)

            hidden_states = torch.stack(latents, dim=-2) # (B, T, n_heads_to_use, D)
            hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states and not self.custom_backward:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attns
        )


class MTPTransformerForCausalLM(MTPTransformerPreTrainedModel, GenerationMixin):

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = MTPTransformerModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None
        self.pad_token_id = config.pad_token_id

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        logits_to_keep: Optional[int] = None,
        **kwargs
    ):
        # only last token for `inputs_ids` if the `past_key_values` is not empty.
        if past_key_values is not None and len(past_key_values) > 0:
            input_ids = input_ids[:, -1:]
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and len(past_key_values) == 0:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard.
            # Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {'input_ids': input_ids.contiguous()}

        if logits_to_keep is not None:
            model_inputs['logits_to_keep'] = logits_to_keep

        model_inputs.update({
            'past_key_values': past_key_values,
            'use_cache': use_cache,
            'attention_mask': attention_mask,
        })
        return model_inputs

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Optional[int] = 0,
        **kwargs: Unpack[Any]
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            return_all_heads=self.training,
            **kwargs
        )

        hidden_states = outputs[0] # (B, T, n_heads_to_use, D)

        all_logits = None
        if not self.training:
            if hidden_states.dim() == 4:
                hidden_states = hidden_states.squeeze(-2) # Remove the n_heads_to_use dimension if not training
            all_logits = self.lm_head(hidden_states)
        else:
            fuse_linear_and_cross_entropy = self.config.fuse_cross_entropy and self.training
            use_custom_backward = self.config.use_custom_backward and self.training
            if use_custom_backward:
                all_logits = SequentialHeadsCustomBackward.apply(
                    hidden_states, self.lm_head, self.model.norm, logits_to_keep, *self.model.extra_heads
                )
            else:
                all_logits = None if fuse_linear_and_cross_entropy else self.lm_head(hidden_states[:, -logits_to_keep:])

        loss = None
        if labels is not None:
            B, T, n_heads_prediction, D = hidden_states.shape
            loss = torch.zeros(1, device=hidden_states.device)
            ntp_loss = torch.zeros(1, device=hidden_states.device)
            mtp_loss = torch.zeros(1, device=hidden_states.device)
            if getattr(self, 'criterion', None) is None:
                if fuse_linear_and_cross_entropy:
                    criterion = FusedLinearCrossEntropyLoss()
                elif self.config.fuse_cross_entropy:
                    criterion = FusedCrossEntropyLoss(inplace_backward=True)
                else:
                    criterion = nn.CrossEntropyLoss()
            else:
                criterion = self.criterion
            # Enable model parallelism
            labels = labels.to(hidden_states.device)
            all_labels = seq_to_mtp(labels, n_future_tokens=n_heads_prediction, model_seq_len=T)
            # Loop across prediction heads
            for i in range(n_heads_prediction):
                # labels in the shape of (B, n_heads_prediction, T)
                labels = all_labels[:, i, :]
                if fuse_linear_and_cross_entropy:
                    current_loss = criterion(hidden_states[:, :, i, :], labels.contiguous(), self.lm_head.weight, self.lm_head.bias)
                else:
                    logits = all_logits[:, :, i, :]
                    current_loss = criterion(logits.view(labels.numel(), -1), labels.reshape(-1))
                if i == 0: # NTP
                    ntp_loss = current_loss
                else:
                    mtp_loss += current_loss
                loss += current_loss

        if not return_dict:
            output = (all_logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return MTPLMOutputWithPast(
            loss=loss,
            ntp_loss=ntp_loss if loss is not None else None,
            mtp_loss=mtp_loss if loss is not None else None,
            logits=all_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
