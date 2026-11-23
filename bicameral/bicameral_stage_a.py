"""Stage A BicameralTriNet built strictly from `trinet.modeling_trinet.TriNetLayer`.

Controlled switches for Stage A ablations:
  - `symmetric=True/False`: when False, changes only Stack A's retention floor
    (`slow_min_retention` vs `min_retention`); Stack B keeps `min_retention`.
  - `callosum=True/False`: when True, exchanges narrow `callosum_dim` summaries
    between layers and learns a per-token output weighting (`_last_alpha_a`);
    when False, joins the two stacks only at the output.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

try:
    from transformers.cache_utils import Cache as _HFCacheBase
except Exception:
    _HFCacheBase = object

from trinet.configuration_trinet import TriNetConfig
from trinet.modeling_trinet import TriNetCache, TriNetLayer


class BicameralTriNetConfig(PretrainedConfig):
    model_type = "bicameral_trinet"
    attribute_map = {"num_hidden_layers": "n_layers"}

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 64,
        d_hemi: Optional[int] = None,
        d_key: int = 32,
        d_val: int = 32,
        n_layers: int = 2,
        delay_lags: int = 4,
        n_receptors: int = 8,
        callosum: bool = True,
        callosum_dim: int = 16,
        symmetric: bool = True,
        min_retention: float = 0.5,
        slow_min_retention: float = 0.95,
        hemisphere_dropout: float = 0.1,
        max_fast_weight: float = 6.0,
        max_write_rate: float = 1.5,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        pad_token_id: int = 0,
        **kwargs,
    ):
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.d_hemi = int(d_model if d_hemi is None else d_hemi)
        self.d_key = int(d_key)
        self.d_val = int(d_val)
        self.n_layers = int(n_layers)
        self.delay_lags = int(delay_lags)
        self.n_receptors = int(n_receptors)
        self.callosum = bool(callosum)
        self.callosum_dim = int(callosum_dim)
        self.symmetric = bool(symmetric)
        self.min_retention = float(min_retention)
        self.slow_min_retention = float(slow_min_retention)
        self.hemisphere_dropout = float(hemisphere_dropout)
        self.max_fast_weight = float(max_fast_weight)
        self.max_write_rate = float(max_write_rate)
        self.use_cache = bool(use_cache)

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def stack_config(self, slow: bool = False) -> TriNetConfig:
        return TriNetConfig(
            vocab_size=self.vocab_size,
            d_model=self.d_hemi,
            d_key=self.d_key,
            d_val=self.d_val,
            n_layers=self.n_layers,
            delay_lags=self.delay_lags,
            n_receptors=self.n_receptors,
            max_fast_weight=self.max_fast_weight,
            max_write_rate=self.max_write_rate,
            min_retention=self.slow_min_retention if slow else self.min_retention,
            use_cache=self.use_cache,
            tie_word_embeddings=self.tie_word_embeddings,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
        )


class BicameralTriNetCache(_HFCacheBase):
    """Constant-size O(1) cache wrapping the two stacks' `TriNetCache` states."""

    is_compileable = False

    def __init__(
        self,
        config: BicameralTriNetConfig,
        batch_size: int,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ):
        if _HFCacheBase is not object:
            try:
                super().__init__()
            except (TypeError, ValueError):
                pass
        self.config = config
        self.num_layers = config.n_layers
        self._seen_tokens = 0
        self.cache_a = TriNetCache(
            config.stack_config(slow=not config.symmetric), batch_size, dtype=dtype, device=device
        )
        self.cache_b = TriNetCache(
            config.stack_config(slow=False), batch_size, dtype=dtype, device=device
        )

    def __len__(self) -> int:
        return self.num_layers

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self._seen_tokens

    def get_max_cache_shape(self) -> Optional[int]:
        return None

    def get_max_length(self) -> Optional[int]:
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        self.cache_a.reorder_cache(beam_idx)
        self.cache_b.reorder_cache(beam_idx)

    def batch_repeat_interleave(self, repeats: int) -> None:
        self.cache_a.batch_repeat_interleave(repeats)
        self.cache_b.batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        self.reorder_cache(indices)

    def reset(self) -> None:
        self._seen_tokens = 0
        self.cache_a.reset()
        self.cache_b.reset()

    def memory_bytes(self) -> int:
        return self.cache_a.memory_bytes() + self.cache_b.memory_bytes()


class CallosumBridge(nn.Module):
    """Narrow inter-layer commissure exchanging compressed summaries between stacks."""

    def __init__(self, d_hemi: int, callosum_dim: int):
        super().__init__()
        self.to_sum_a = nn.Linear(d_hemi, callosum_dim)
        self.to_sum_b = nn.Linear(d_hemi, callosum_dim)
        self.from_b = nn.Linear(callosum_dim, d_hemi)
        self.from_a = nn.Linear(callosum_dim, d_hemi)
        self.norm_a = nn.LayerNorm(d_hemi)
        self.norm_b = nn.LayerNorm(d_hemi)

    def forward(self, h_a: torch.Tensor, h_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sum_a = F.silu(self.to_sum_a(h_a))
        sum_b = F.silu(self.to_sum_b(h_b))
        return self.norm_a(h_a + self.from_b(sum_b)), self.norm_b(h_b + self.from_a(sum_a))


class OutputCallosum(nn.Module):
    """Narrow output commissure learning a per-token weighting over the two stacks."""

    def __init__(self, d_hemi: int, callosum_dim: int, hemisphere_dropout: float):
        super().__init__()
        self.hemisphere_dropout = float(hemisphere_dropout)
        self.to_sum_a = nn.Linear(d_hemi, callosum_dim)
        self.to_sum_b = nn.Linear(d_hemi, callosum_dim)
        self.gate = nn.Linear(3 * callosum_dim, 1)

    def forward(self, h_a: torch.Tensor, h_b: torch.Tensor) -> torch.Tensor:
        sum_a = F.silu(self.to_sum_a(h_a))
        sum_b = F.silu(self.to_sum_b(h_b))
        alpha_a = torch.sigmoid(self.gate(torch.cat([sum_a, sum_b, (sum_a - sum_b).abs()], dim=-1)))
        if self.training and self.hemisphere_dropout > 0.0:
            u = torch.rand(h_a.shape[0], 1, 1, device=h_a.device, dtype=h_a.dtype)
            drop_a = u < (0.5 * self.hemisphere_dropout)
            drop_b = (u >= (0.5 * self.hemisphere_dropout)) & (u < self.hemisphere_dropout)
            alpha_a = torch.where(
                drop_a, torch.zeros_like(alpha_a), torch.where(drop_b, torch.ones_like(alpha_a), alpha_a)
            )
        return alpha_a


class BicameralTriNetPreTrainedModel(PreTrainedModel):
    config_class = BicameralTriNetConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["TriNetLayer"]

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d, nn.LayerNorm)):
            module.reset_parameters()
        elif isinstance(module, TriNetLayer):
            nn.init.constant_(module.yellow_to_write.bias, 0.25)
            nn.init.constant_(module.yellow_to_retain.bias, 0.90)
        elif isinstance(module, OutputCallosum):
            nn.init.zeros_(module.gate.bias)


class BicameralTriNetModel(BicameralTriNetPreTrainedModel):
    def __init__(self, config: BicameralTriNetConfig):
        super().__init__(config)
        cfg_a = config.stack_config(slow=not config.symmetric)
        cfg_b = config.stack_config(slow=False)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.in_a = nn.Linear(config.d_model, config.d_hemi)
        self.in_b = nn.Linear(config.d_model, config.d_hemi)
        self.layers_a = nn.ModuleList([TriNetLayer(cfg_a, layer_idx=i) for i in range(config.n_layers)])
        self.layers_b = nn.ModuleList([TriNetLayer(cfg_b, layer_idx=i) for i in range(config.n_layers)])

        if config.callosum:
            self.callosum_bridges = nn.ModuleList(
                [CallosumBridge(config.d_hemi, config.callosum_dim) for _ in range(max(0, config.n_layers - 1))]
            )
            self.out_callosum = OutputCallosum(config.d_hemi, config.callosum_dim, config.hemisphere_dropout)
        else:
            self.callosum_bridges = None
            self.out_callosum = None

        self.out_a = nn.Linear(config.d_hemi, config.d_model)
        self.out_b = nn.Linear(config.d_hemi, config.d_model)
        self.final_norm = nn.LayerNorm(config.d_model)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if past_key_values is None:
            past_key_values = kwargs.get("cache")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of `input_ids` or `inputs_embeds`.")

        emb = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        bsz, seq_len, _ = emb.shape

        if isinstance(past_key_values, BicameralTriNetCache):
            cache: Optional[BicameralTriNetCache] = past_key_values
        elif use_cache:
            cache = BicameralTriNetCache(self.config, batch_size=bsz, dtype=emb.dtype, device=emb.device)
        else:
            cache = None

        h_a, h_b = self.in_a(emb), self.in_b(emb)
        all_hidden = (emb,) if output_hidden_states else None
        eta_acc = emb.new_zeros(())

        for idx in range(self.config.n_layers):
            h_a, fa, ya, da, eta_a = self.layers_a[idx](
                h_a,
                fast_weights=cache.cache_a.fast_weights[idx] if cache is not None else None,
                yellow_tonic=cache.cache_a.yellow_tonic[idx] if cache is not None else None,
                delay_buffer=cache.cache_a.delay_buffers[idx] if cache is not None else None,
            )
            h_b, fb, yb, db, eta_b = self.layers_b[idx](
                h_b,
                fast_weights=cache.cache_b.fast_weights[idx] if cache is not None else None,
                yellow_tonic=cache.cache_b.yellow_tonic[idx] if cache is not None else None,
                delay_buffer=cache.cache_b.delay_buffers[idx] if cache is not None else None,
            )
            eta_acc = eta_acc + 0.5 * (eta_a + eta_b)

            if cache is not None:
                cache.cache_a.fast_weights[idx], cache.cache_a.yellow_tonic[idx], cache.cache_a.delay_buffers[idx] = fa, ya, da
                cache.cache_b.fast_weights[idx], cache.cache_b.yellow_tonic[idx], cache.cache_b.delay_buffers[idx] = fb, yb, db

            if self.callosum_bridges is not None and idx < self.config.n_layers - 1:
                h_a, h_b = self.callosum_bridges[idx](h_a, h_b)

            if output_hidden_states:
                all_hidden = all_hidden + (torch.cat([h_a, h_b], dim=-1),)

        if self.out_callosum is not None:
            alpha_a = self.out_callosum(h_a, h_b)
            h_out = self.final_norm(alpha_a * self.out_a(h_a) + (1.0 - alpha_a) * self.out_b(h_b))
            self._last_alpha_a = alpha_a
        else:
            h_out = self.final_norm(self.out_a(h_a) + self.out_b(h_b))
            self._last_alpha_a = None

        if output_hidden_states:
            all_hidden = all_hidden + (h_out,)
        if cache is not None:
            cache._seen_tokens += seq_len
        self._last_mean_eta = eta_acc / max(1, self.config.n_layers)

        if not return_dict:
            return tuple(v for v in (h_out, cache, all_hidden) if v is not None)
        return BaseModelOutputWithPast(last_hidden_state=h_out, past_key_values=cache, hidden_states=all_hidden)


class BicameralTriNetForCausalLM(BicameralTriNetPreTrainedModel, GenerationMixin):
    def __init__(self, config: BicameralTriNetConfig):
        super().__init__(config)
        self.model = BicameralTriNetModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head = new_embeddings

    def get_decoder(self) -> BicameralTriNetModel:
        return self.model

    def set_decoder(self, decoder: BicameralTriNetModel) -> None:
        self.model = decoder

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = True,
        **kwargs,
    ) -> dict:
        if past_key_values is None:
            past_key_values = kwargs.get("cache")
        if isinstance(past_key_values, BicameralTriNetCache) and past_key_values.get_seq_length() > 0:
            input_ids = input_ids[:, -1:]
            inputs_embeds = None
        model_inputs: dict = {"inputs_embeds": inputs_embeds} if inputs_embeds is not None else {"input_ids": input_ids}
        model_inputs["past_key_values"] = past_key_values
        model_inputs["use_cache"] = use_cache
        model_inputs["attention_mask"] = attention_mask
        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Any] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        metabolic_weight: float = 0.0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if past_key_values is None:
            past_key_values = kwargs.get("cache")
        return_dict = self.config.use_return_dict if return_dict is None else return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        logits = self.lm_head(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().to(logits.device).view(-1),
                ignore_index=-100,
            )
            if metabolic_weight > 0.0:
                loss = loss + metabolic_weight * self.model._last_mean_eta

        if not return_dict:
            output = (logits, outputs.past_key_values)
            return ((loss,) + output) if loss is not None else output
        return CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states
        )


BicameralTriNetConfig.register_for_auto_class()
BicameralTriNetModel.register_for_auto_class("AutoModel")
BicameralTriNetForCausalLM.register_for_auto_class("AutoModelForCausalLM")
