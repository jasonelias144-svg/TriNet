"""Hugging Face Transformers implementation of the 6-archetype TriNet architecture.

Provides O(1) constant-memory stateful generation via `TriNetCache`, holding each
layer's fixed-size dissolving fast weights `F_fast`, tonic Yellow modulator state,
and K-tap causal `DelayLineUnit` buffer.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

try:
    from transformers.cache_utils import Cache as _HFCacheBase
except Exception:
    _HFCacheBase = object

from .configuration_trinet import TriNetConfig


class TriNetCache(_HFCacheBase):
    """
    Fixed-size O(1) recurrent state cache for TriNet generation.

    Unlike Transformer KV caches that grow O(T) with sequence length, every tensor
    stored in `TriNetCache` stays strictly constant in shape regardless of how many
    tokens have been processed:
      - fast_weights[l]:  (B, d_key, d_val)
      - yellow_tonic[l]:  (B, 1)
      - delay_buffers[l]: (B, d_model, delay_lags - 1)
    """

    is_compileable = False

    def __init__(
        self,
        config: TriNetConfig,
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
        self.fast_weights: List[torch.Tensor] = [
            torch.zeros(batch_size, config.d_key, config.d_val, dtype=dtype, device=device)
            for _ in range(config.n_layers)
        ]
        self.yellow_tonic: List[torch.Tensor] = [
            torch.zeros(batch_size, 1, dtype=dtype, device=device)
            for _ in range(config.n_layers)
        ]
        buf_len = max(0, config.delay_lags - 1)
        self.delay_buffers: List[torch.Tensor] = [
            torch.zeros(batch_size, config.d_model, buf_len, dtype=dtype, device=device)
            for _ in range(config.n_layers)
        ]

    def __len__(self) -> int:
        return self.num_layers

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self._seen_tokens

    def get_max_cache_shape(self) -> Optional[int]:
        return None

    def get_max_length(self) -> Optional[int]:
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        for i in range(self.num_layers):
            idx = beam_idx.to(self.fast_weights[i].device)
            self.fast_weights[i] = self.fast_weights[i].index_select(0, idx)
            self.yellow_tonic[i] = self.yellow_tonic[i].index_select(0, idx)
            self.delay_buffers[i] = self.delay_buffers[i].index_select(0, idx)

    def batch_repeat_interleave(self, repeats: int) -> None:
        for i in range(self.num_layers):
            self.fast_weights[i] = self.fast_weights[i].repeat_interleave(repeats, dim=0)
            self.yellow_tonic[i] = self.yellow_tonic[i].repeat_interleave(repeats, dim=0)
            self.delay_buffers[i] = self.delay_buffers[i].repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        self.reorder_cache(indices)

    def reset(self) -> None:
        self._seen_tokens = 0
        for i in range(self.num_layers):
            self.fast_weights[i].zero_()
            self.yellow_tonic[i].zero_()
            self.delay_buffers[i].zero_()

    def memory_bytes(self) -> int:
        total = 0
        for i in range(self.num_layers):
            total += self.fast_weights[i].numel() * self.fast_weights[i].element_size()
            total += self.yellow_tonic[i].numel() * self.yellow_tonic[i].element_size()
            total += self.delay_buffers[i].numel() * self.delay_buffers[i].element_size()
        return total


class ReceptorBroadcastBus(nn.Module):
    """Archetype 5: 3-color composite hue emitter + Michaelis-Menten hue receptors."""

    def __init__(self, d_in: int, n_receptors: int):
        super().__init__()
        self.to_rby = nn.Linear(d_in, 3, bias=True)
        self.affinity_logits = nn.Parameter(torch.randn(n_receptors, 3) * 0.5)
        self.log_sigma = nn.Parameter(torch.full((n_receptors,), math.log(0.35)))
        self.log_g0 = nn.Parameter(torch.tensor(math.log(0.05)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        field_rby = F.relu(self.to_rby(x))
        intensity = field_rby.sum(dim=-1, keepdim=True)
        hue = field_rby / (intensity + 1e-6)

        affinities = F.softmax(self.affinity_logits, dim=-1)  # (M, 3)
        s2 = torch.exp(2.0 * self.log_sigma.clamp(-3.0, 1.0))  # (M,)
        g0 = torch.exp(self.log_g0.clamp(-5.0, 0.0))

        diff = hue.unsqueeze(2) - affinities.view(1, 1, -1, 3)
        dist_sq = (diff * diff).sum(dim=-1)
        hue_match = s2 / (s2 + dist_sq)
        shunted_int = intensity / (g0 + intensity)
        return hue_match * shunted_int


class TriNetLayer(nn.Module):
    """
    One stackable 6-archetype heterogeneous layer:
      - DelayLineUnit:        causal K-tap conv with O(1) shift buffer
      - ReceptorBroadcastBus: 3-color chromaticity routing into the Yellow modulator
      - TriChannel memory:    W_base + dissolving F_fast (O(1) synaptic matrix)
      - ShuntingInterneuron:  Blue divisive normalization over Red readout
      - PyramidalComparator:  basal value vs apical F_fast mismatch (surprise)
      - ModulatorNucleus:     slow tonic Yellow + sparse dead-zone write/retain gates
    """

    def __init__(self, config: TriNetConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_model = config.d_model
        self.d_key = config.d_key
        self.d_val = config.d_val
        self.delay_lags = config.delay_lags
        self.max_fast_weight = config.max_fast_weight
        self.max_write_rate = config.max_write_rate
        self.min_retention = config.min_retention

        # Archetype 6: Causal DelayLineUnit over token representations
        self.delay_conv = nn.Conv1d(
            config.d_model, config.d_model, kernel_size=config.delay_lags, padding=0
        )

        # Archetype 5: Composite-color ReceptorBroadcastBus
        self.receptor_bus = ReceptorBroadcastBus(config.d_model, config.n_receptors)

        # Separate write-key, read-query, and basal-value projections
        self.W_key = nn.Linear(config.d_model, config.d_key, bias=False)
        self.W_query = nn.Linear(config.d_model, config.d_key, bias=False)
        self.W_val = nn.Linear(config.d_model, config.d_val, bias=False)

        # Archetype 1: Slow baseline weights W_base
        self.W_base = nn.Parameter(
            torch.randn(config.d_key, config.d_val) * (1.0 / math.sqrt(config.d_key))
        )

        # Archetype 3: Blue ShuntingInterneuron pool
        self.shunt_w = nn.Parameter(torch.full((config.d_val,), 0.15))
        self.log_g0 = nn.Parameter(torch.tensor(math.log(0.25)))

        # Archetypes 2 & 4: Pyramidal surprise + ReceptorBroadcast -> ModulatorNucleus
        mod_in_dim = config.n_receptors + 1
        gate_in_dim = config.n_receptors + 2
        self.tonic_decay_logit = nn.Parameter(torch.tensor(1.0))
        self.mod_in = nn.Linear(mod_in_dim, 1)
        self.yellow_to_write = nn.Linear(gate_in_dim, 1)
        self.yellow_to_retain = nn.Linear(gate_in_dim, 1)

        # Residual fusion
        self.out_proj = nn.Linear(
            config.d_val + config.d_model + config.n_receptors, config.d_model
        )
        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        fast_weights: Optional[torch.Tensor] = None,
        yellow_tonic: Optional[torch.Tensor] = None,
        delay_buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        # 1. DelayLineUnit causal shift-register convolution (exact O(1) incremental state)
        h_t = hidden_states.transpose(1, 2)  # (B, d_model, T)
        buf_len = self.delay_lags - 1
        if buf_len > 0:
            if delay_buffer is None:
                delay_buffer = torch.zeros(bsz, self.d_model, buf_len, device=device, dtype=dtype)
            h_cat = torch.cat([delay_buffer, h_t], dim=-1)
            next_delay_buffer = h_cat[:, :, -buf_len:].contiguous()
        else:
            h_cat = h_t
            next_delay_buffer = torch.zeros(bsz, self.d_model, 0, device=device, dtype=dtype)

        d_ctx = F.silu(self.delay_conv(h_cat).transpose(1, 2))  # (B, T, d_model)

        # 2. Composite-color ReceptorBroadcastBus over local syntax context
        rec_feat = self.receptor_bus(d_ctx)  # (B, T, n_receptors)

        # 3. Normalized write-keys, read-queries, and basal values
        k_write = F.silu(self.W_key(d_ctx))
        k_write = k_write / torch.sqrt((k_write * k_write).sum(dim=-1, keepdim=True) + 1e-4)

        q_read = F.silu(self.W_query(d_ctx))
        q_read = q_read / torch.sqrt((q_read * q_read).sum(dim=-1, keepdim=True) + 1e-4)

        v_basal_all = self.W_val(hidden_states)  # (B, T, d_val)

        # 4. Recurrent dissolving fast-weight & ModulatorNucleus loop
        if fast_weights is None:
            f_fast = torch.zeros(bsz, self.d_key, self.d_val, device=device, dtype=dtype)
        else:
            f_fast = fast_weights

        if yellow_tonic is None:
            y_tonic = torch.zeros(bsz, 1, device=device, dtype=dtype)
        else:
            y_tonic = yellow_tonic

        tonic_decay = torch.sigmoid(self.tonic_decay_logit)
        g0 = torch.exp(self.log_g0.clamp(-4.0, 1.0))
        shunt_w = F.softplus(self.shunt_w)
        retain_span = 1.0 - self.min_retention

        mem_outs = []
        eta_sum = hidden_states.new_zeros(())

        for t in range(seq_len):
            qt = q_read[:, t, :]
            kt = k_write[:, t, :]
            vt = v_basal_all[:, t, :]
            rt = rec_feat[:, t, :]

            # Associative readout through W_base + F_fast under Blue shunting
            r_read = torch.matmul(qt, self.W_base) + torch.bmm(qt.unsqueeze(1), f_fast).squeeze(1)
            b_shunt = (F.relu(r_read) * shunt_w).sum(dim=-1, keepdim=True)
            mem_outs.append(r_read / (g0 + b_shunt))

            # Pyramidal mismatch (basal value vs apical fast-memory prediction)
            v_apical = torch.bmm(kt.unsqueeze(1), f_fast).squeeze(1)
            delta_err = vt - v_apical
            surprise = torch.sqrt((delta_err * delta_err).mean(dim=-1, keepdim=True) + 1e-6)

            # ModulatorNucleus slow Yellow + sparse dead-zone write/retain gates
            mod_drive = F.relu(self.mod_in(torch.cat([rt, surprise], dim=-1)))
            y_tonic = torch.clamp(tonic_decay * y_tonic + mod_drive, 0.0, 2.0)

            gate_in = torch.cat([rt, surprise, y_tonic], dim=-1)
            eta = self.max_write_rate * torch.clamp(self.yellow_to_write(gate_in), 0.0, 1.0)
            lam = self.min_retention + retain_span * torch.clamp(
                self.yellow_to_retain(gate_in), 0.0, 1.0
            )
            eta_sum = eta_sum + eta.mean()

            outer = torch.bmm(kt.unsqueeze(2), delta_err.unsqueeze(1))
            f_fast = torch.clamp(
                lam.unsqueeze(-1) * f_fast + eta.unsqueeze(-1) * outer,
                -self.max_fast_weight,
                self.max_fast_weight,
            )

        mem_seq = torch.stack(mem_outs, dim=1)
        fused = torch.cat([mem_seq, d_ctx, rec_feat], dim=-1)
        next_hidden = self.norm(hidden_states + self.out_proj(fused))
        return next_hidden, f_fast, y_tonic, next_delay_buffer, eta_sum / max(1, seq_len)


class TriNetPreTrainedModel(PreTrainedModel):
    config_class = TriNetConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["TriNetLayer"]

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d, nn.LayerNorm)):
            module.reset_parameters()  # PyTorch's own initial values (provenance/MANIFEST.json, change 5)
        if not isinstance(module, TriNetLayer):
            return
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.Conv1d):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, TriNetLayer):
            # Keep dead-zone gates inside their active gradient window at step 0.
            nn.init.constant_(module.yellow_to_write.bias, 0.25)
            nn.init.constant_(module.yellow_to_retain.bias, 0.90)


class TriNetModel(TriNetPreTrainedModel):
    """Bare TriNet backbone outputting hidden states and O(1) `TriNetCache`."""

    def __init__(self, config: TriNetConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            [TriNetLayer(config, layer_idx=i) for i in range(config.n_layers)]
        )
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
        use_cache = self.config.use_cache if use_cache is None else use_cache
        return_dict = self.config.use_return_dict if return_dict is None else return_dict

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of `input_ids` or `inputs_embeds`.")

        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds

        bsz, seq_len, _ = hidden_states.shape

        # Accept either a live TriNetCache or initialize a fresh O(1) cache
        # (transparently replacing an empty DynamicCache if generate() created one).
        cache: Optional[TriNetCache]
        if isinstance(past_key_values, TriNetCache):
            cache = past_key_values
        elif use_cache:
            cache = TriNetCache(
                self.config, batch_size=bsz, dtype=hidden_states.dtype, device=hidden_states.device
            )
        else:
            cache = None

        all_hidden_states = () if output_hidden_states else None
        eta_acc = hidden_states.new_zeros(())

        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            f_in = cache.fast_weights[idx] if cache is not None else None
            y_in = cache.yellow_tonic[idx] if cache is not None else None
            d_in = cache.delay_buffers[idx] if cache is not None else None

            hidden_states, f_out, y_out, d_out, mean_eta = layer(
                hidden_states,
                fast_weights=f_in,
                yellow_tonic=y_in,
                delay_buffer=d_in,
            )
            eta_acc = eta_acc + mean_eta

            if cache is not None:
                cache.fast_weights[idx] = f_out
                cache.yellow_tonic[idx] = y_out
                cache.delay_buffers[idx] = d_out

        hidden_states = self.final_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if cache is not None:
            cache._seen_tokens += seq_len

        self._last_mean_eta = eta_acc / max(1, len(self.layers))

        if not return_dict:
            return tuple(v for v in (hidden_states, cache, all_hidden_states) if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=cache,
            hidden_states=all_hidden_states,
        )


class TriNetForCausalLM(TriNetPreTrainedModel, GenerationMixin):
    """Causal language model head over `TriNetModel` with O(1) stateful generation."""

    def __init__(self, config: TriNetConfig):
        super().__init__(config)
        self.model = TriNetModel(config)
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

    def get_decoder(self) -> TriNetModel:
        return self.model

    def set_decoder(self, decoder: TriNetModel) -> None:
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
        if isinstance(past_key_values, TriNetCache) and past_key_values.get_seq_length() > 0:
            input_ids = input_ids[:, -1:]
            inputs_embeds = None

        model_inputs: dict = {}
        if inputs_embeds is not None:
            model_inputs["inputs_embeds"] = inputs_embeds
        else:
            model_inputs["input_ids"] = input_ids

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
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous().to(logits.device)
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            if metabolic_weight > 0.0:
                loss = loss + metabolic_weight * self.model._last_mean_eta

        if not return_dict:
            output = (logits, outputs.past_key_values)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )


TriNetModel.register_for_auto_class("AutoModel")
TriNetForCausalLM.register_for_auto_class("AutoModelForCausalLM")
