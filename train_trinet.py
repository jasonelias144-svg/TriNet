"""Byte-level autoregressive training and evaluation harness for TriNetForCausalLM
versus a parameter-matched Causal Transformer baseline.

Supports:
  - Hugging Face streaming datasets (e.g. `roneneldan/TinyStories`, `HuggingFaceFW/fineweb-edu`)
  - Local UTF-8 text corpora (`--local-text path/to/corpus.txt`)
  - Offline `--smoke-test` mode for fast end-to-end verification
  - Configurable `--reasoning-mix` blending dynamic in-context binding/overwrite/2-hop tasks
  - Direct `save_pretrained(..., safe_serialization=True)` and optional `--push-to-hub`
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from trinet.configuration_trinet import TriNetConfig
from trinet.modeling_trinet import TriNetForCausalLM


# ==============================================================================
# 1. LOSSLESS BYTE-LEVEL UTF-8 TOKENIZER (VOCAB = 256)
# ==============================================================================

class ByteTokenizer:
    """
    Lossless byte-level UTF-8 tokenizer aligned with `TriNetConfig` defaults:
      - 0 (0x00 NUL): pad_token_id
      - 2 (0x02 STX): bos_token_id
      - 3 (0x03 ETX): eos_token_id
      - Every raw UTF-8 byte maps 1:1 to its integer value in [0, 255].
    """

    vocab_size: int = 256
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int = 3

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = list(text.encode("utf-8"))
        if add_bos:
            ids = [self.bos_token_id] + ids
        if add_eos:
            ids = ids + [self.eos_token_id]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        specials = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        raw = bytes(int(i) & 0xFF for i in ids if not (skip_special_tokens and i in specials))
        return raw.decode("utf-8", errors="replace")

    def save_metadata(self, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        meta = {
            "tokenizer_type": "byte_utf8",
            "vocab_size": self.vocab_size,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
        }
        with open(os.path.join(save_dir, "byte_tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


# ==============================================================================
# 2. PARAMETER-MATCHED CAUSAL TRANSFORMER BASELINE
# ==============================================================================

class MatchedTransformerLM(nn.Module):
    """
    Standard pre-norm Causal Transformer automatically sized to match the parameter
    count of `TriNetForCausalLM` so comparisons hold parameter budget constant.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        max_len: int = 512,
        use_pos_emb: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.use_pos_emb = use_pos_emb
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        if use_pos_emb:
            self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        bsz, seq_len = input_ids.shape
        h = self.embed_tokens(input_ids)
        if self.use_pos_emb:
            pos = torch.arange(seq_len, device=input_ids.device).clamp(max=self.max_len - 1)
            h = h + self.pos_emb(pos.unsqueeze(0))
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool), diagonal=1
        )
        h = self.final_norm(self.encoder(h, mask=mask))
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, self.vocab_size),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return loss, logits

    @torch.no_grad()
    def generate_greedy(
        self, input_ids: torch.LongTensor, max_new_tokens: int = 32
    ) -> torch.LongTensor:
        cur = input_ids.clone()
        for _ in range(max_new_tokens):
            _, logits = self.forward(cur)
            nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            cur = torch.cat([cur, nxt], dim=-1)
        return cur


def build_matched_transformer(
    target_params: int,
    config: TriNetConfig,
    use_pos_emb: bool = True,
    max_len: int = 512,
) -> MatchedTransformerLM:
    """Solves for `d_ff` so `MatchedTransformerLM` matches `target_params` tightly."""
    d_model = config.d_model
    n_layers = config.n_layers
    n_heads = max(1, min(8, d_model // 16))
    while d_model % n_heads != 0 and n_heads > 1:
        n_heads -= 1

    m_lo = MatchedTransformerLM(
        vocab_size=config.vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=16,
        max_len=max_len,
        use_pos_emb=use_pos_emb,
    )
    m_hi = MatchedTransformerLM(
        vocab_size=config.vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=32,
        max_len=max_len,
        use_pos_emb=use_pos_emb,
    )
    p_lo = sum(p.numel() for p in m_lo.parameters())
    p_hi = sum(p.numel() for p in m_hi.parameters())
    slope = max(1, (p_hi - p_lo) // 16)
    best_dff = max(16, int(round(16 + (target_params - p_lo) / slope)))
    best_dff = max(16, (best_dff // 8) * 8)

    return MatchedTransformerLM(
        vocab_size=config.vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=best_dff,
        max_len=max_len,
        use_pos_emb=use_pos_emb,
    )


# ==============================================================================
# 3. DATA PIPELINE: STREAMING PROSE + DYNAMIC IN-CONTEXT REASONING MIX
# ==============================================================================

_VARS = list("abcdefgh")
_VALS = list("012345")
_FILLER = list("xyz .,-")

_SMOKE_PROSE = (
    "In a quiet workshop by the river, two builders compared their maps before dawn. "
    "One traced every path with a silver compass, while the other measured how water moved "
    "through narrow wooden gates. Whenever a pattern shifted, they kept the earlier sketch "
    "beside the new one so nothing learned was lost. "
    "A relation requires a carrier, a respect, and a clear boundary. When two voices answer "
    "across time, repetition builds memory while variation reveals what stayed invariant. "
)


def make_reasoning_example(
    n_direct: int = 3,
    l1: int = 6,
    l2: int = 6,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[str, List[Tuple[int, int, int]]]:
    """
    Generates one UTF-8 dynamic reasoning episode:
      direct bindings -> filler 1 -> 2-hop pointer -> mid-stream overwrite -> filler 2 -> queries.
    Returns `(text, query_meta)` where each tuple is:
      `(pos_of_gt_char, expected_digit_byte_id, kind_code)` with 1=direct, 2=overwrite, 3=2-hop.
    """
    rng = rng or np.random.default_rng()
    chosen = rng.choice(_VARS, size=n_direct + 1, replace=False)
    direct_vars = list(chosen[:n_direct])
    ptr_var = str(chosen[n_direct])

    env: Dict[str, str] = {}
    parts: List[str] = []
    for v in direct_vars:
        d = str(int(rng.integers(0, len(_VALS))))
        env[v] = d
        parts.append(f"{v}={d};")

    f1 = "".join(
        str(rng.choice(_VARS)) if rng.random() < 0.18 else str(rng.choice(_FILLER))
        for _ in range(l1)
    )
    parts.append(f1)

    # Change 6 (provenance/MANIFEST.json): which variable the pointer names,
    # and which is overwritten, are drawn at random, not always the first two.
    src_i, reb_i = rng.choice(n_direct, size=2, replace=False)
    src_var = direct_vars[src_i]
    parts.append(f"{ptr_var}={src_var};")

    reb_var = direct_vars[reb_i]
    new_d = str((int(env[reb_var]) + int(rng.integers(1, len(_VALS)))) % len(_VALS))
    env[reb_var] = new_d
    parts.append(f"{reb_var}={new_d};")

    f2 = "".join(
        str(rng.choice(_VARS)) if rng.random() < 0.18 else str(rng.choice(_FILLER))
        for _ in range(l2)
    )
    parts.append(f2)

    q_vars = direct_vars + [ptr_var]
    rng.shuffle(q_vars)

    text_chars = list("".join(parts))
    query_meta: List[Tuple[int, int, int]] = []
    for qv in q_vars:
        ans = env[src_var] if qv == ptr_var else env[qv]
        kind = 3 if qv == ptr_var else (2 if qv == reb_var else 1)
        text_chars.extend(["?", qv, ">"])
        gt_pos = len(text_chars) - 1
        query_meta.append((gt_pos, ord(ans), kind))
        text_chars.extend([ans, ";"])

    return "".join(text_chars), query_meta


def reasoning_batch_for_eval(
    batch_size: int,
    l1: int,
    l2: int,
    tokenizer: ByteTokenizer,
    rng: np.random.Generator,
    device: torch.device,
) -> Tuple[torch.LongTensor, List[List[Tuple[int, int, int]]]]:
    seqs: List[List[int]] = []
    metas: List[List[Tuple[int, int, int]]] = []
    for _ in range(batch_size):
        text, meta = make_reasoning_example(n_direct=3, l1=l1, l2=l2, rng=rng)
        ids = tokenizer.encode(text, add_bos=True, add_eos=False)
        shifted = [(pos + 1, ans_id, kind) for (pos, ans_id, kind) in meta]
        seqs.append(ids)
        metas.append(shifted)
    max_len = max(len(s) for s in seqs)
    arr = np.full((batch_size, max_len), tokenizer.pad_token_id, dtype=np.int64)
    for i, s in enumerate(seqs):
        arr[i, : len(s)] = s
    return torch.from_numpy(arr).to(device), metas


class ByteStreamProvider:
    """Yields fixed-length `(input_ids, labels)` chunks from HF streaming, local file, or smoke prose."""

    def __init__(
        self,
        tokenizer: ByteTokenizer,
        seq_len: int,
        dataset_name: Optional[str] = None,
        dataset_config: Optional[str] = None,
        dataset_split: str = "train",
        text_column: str = "text",
        local_text_path: Optional[str] = None,
        smoke_test: bool = False,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)
        self.mode = "smoke"
        self.local_bytes: Optional[np.ndarray] = None
        self.hf_iter: Optional[Iterator[str]] = None
        self.buf: List[int] = []

        if local_text_path is not None:
            with open(local_text_path, "rb") as f:
                raw = f.read()
            if len(raw) < seq_len + 2:
                raw = raw * ((seq_len + 2) // max(1, len(raw)) + 2)
            self.local_bytes = np.frombuffer(raw, dtype=np.uint8).astype(np.int64)
            self.mode = "local"
        elif dataset_name is not None and not smoke_test:
            from datasets import load_dataset

            ds = load_dataset(dataset_name, dataset_config, split=dataset_split, streaming=True)
            self._ds = ds
            self._text_column = text_column
            self.hf_iter = self._make_hf_iter()
            self.mode = "hf"
        else:
            raw = (_SMOKE_PROSE * 64).encode("utf-8")
            self.local_bytes = np.frombuffer(raw, dtype=np.uint8).astype(np.int64)
            self.mode = "smoke"

    def _make_hf_iter(self) -> Iterator[str]:
        while True:
            for row in self._ds:
                txt = row.get(self._text_column, "")
                if txt:
                    yield txt

    def sample_prose_chunk(self) -> List[int]:
        if self.mode in ("local", "smoke"):
            assert self.local_bytes is not None
            n = len(self.local_bytes)
            start = int(self.rng.integers(0, n - self.seq_len))
            return self.local_bytes[start : start + self.seq_len].tolist()

        assert self.hf_iter is not None
        while len(self.buf) < self.seq_len:
            txt = next(self.hf_iter)
            self.buf.extend(self.tokenizer.encode(txt, add_bos=True, add_eos=True))
        chunk = self.buf[: self.seq_len]
        self.buf = self.buf[self.seq_len :]
        return chunk

    def sample_mixed_batch(
        self,
        batch_size: int,
        reasoning_mix: float = 0.20,
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        xs = np.full((batch_size, self.seq_len), self.tokenizer.pad_token_id, dtype=np.int64)
        ys = np.full((batch_size, self.seq_len), -100, dtype=np.int64)

        for b in range(batch_size):
            if self.rng.random() < reasoning_mix:
                ids: List[int] = []
                while len(ids) < self.seq_len:
                    l1 = int(self.rng.integers(2, 10))
                    l2 = int(self.rng.integers(2, 10))
                    txt, _ = make_reasoning_example(n_direct=3, l1=l1, l2=l2, rng=self.rng)
                    ids.extend(self.tokenizer.encode(txt, add_bos=True, add_eos=True))
                chunk = ids[: self.seq_len]
            else:
                chunk = self.sample_prose_chunk()

            xs[b, :] = chunk
            ys[b, :] = chunk

        return torch.from_numpy(xs), torch.from_numpy(ys)


# ==============================================================================
# 4. OPTIMIZER, LR SCHEDULE & EVALUATION HARNESS
# ==============================================================================

def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.AdamW:
    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "norm" in name or "bias" in name or "log_" in name or "logit" in name:
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
    )


def get_cosine_lr(
    step: int, total_steps: int, warmup_steps: int, max_lr: float, min_lr: float
) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate_prose_bpb(
    model: nn.Module,
    val_stream: ByteStreamProvider,
    batch_size: int,
    n_batches: int,
    device: torch.device,
    is_trinet: bool,
) -> Dict[str, float]:
    model.eval()
    losses = []
    for _ in range(n_batches):
        xb, yb = val_stream.sample_mixed_batch(batch_size, reasoning_mix=0.0)
        xb, yb = xb.to(device), yb.to(device)
        if is_trinet:
            loss = model(input_ids=xb, labels=yb, use_cache=False).loss
        else:
            loss, _ = model(input_ids=xb, labels=yb)
        losses.append(float(loss.item()))
    mean_nats = float(np.mean(losses))
    return {
        "loss_nats": mean_nats,
        "bpb": mean_nats / math.log(2.0),
        "perplexity": math.exp(min(20.0, mean_nats)),
    }


@torch.no_grad()
def evaluate_reasoning_suite(
    model: nn.Module,
    tokenizer: ByteTokenizer,
    device: torch.device,
    is_trinet: bool,
    filler_lengths: List[int],
    n_batches: int = 4,
    batch_size: int = 24,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    results: Dict[str, Dict[str, float]] = {}
    for filler_len in filler_lengths:
        rng = np.random.default_rng(7000 + filler_len)
        counts = {1: [0, 0], 2: [0, 0], 3: [0, 0]}
        for _ in range(n_batches):
            xb, metas = reasoning_batch_for_eval(
                batch_size, l1=filler_len, l2=filler_len, tokenizer=tokenizer, rng=rng, device=device
            )
            if is_trinet:
                logits = model(input_ids=xb, use_cache=False).logits
            else:
                _, logits = model(input_ids=xb)
            preds = logits.argmax(dim=-1).cpu().numpy()
            for b, meta in enumerate(metas):
                for pos, ans_id, kind in meta:
                    counts[kind][1] += 1
                    if int(preds[b, pos]) == ans_id:
                        counts[kind][0] += 1
        all_c = sum(v[0] for v in counts.values())
        all_n = max(1, sum(v[1] for v in counts.values()))
        results[f"filler_{filler_len}"] = {
            "all_acc": 100.0 * all_c / all_n,
            "direct_acc": 100.0 * counts[1][0] / max(1, counts[1][1]),
            "overwrite_acc": 100.0 * counts[2][0] / max(1, counts[2][1]),
            "two_hop_acc": 100.0 * counts[3][0] / max(1, counts[3][1]),
        }
    return results


@torch.no_grad()
def verify_o1_cache(model: TriNetForCausalLM, device: torch.device) -> Dict[str, float]:
    model.eval()
    probe = torch.randint(32, 122, (1, 16), device=device)
    full_logits = model(input_ids=probe, use_cache=False).logits

    past = None
    chunks = []
    mem_sizes = []
    for t in range(probe.shape[1]):
        out = model(input_ids=probe[:, t : t + 1], past_key_values=past, use_cache=True)
        past = out.past_key_values
        chunks.append(out.logits)
        mem_sizes.append(past.memory_bytes())
    stream_logits = torch.cat(chunks, dim=1)
    max_diff = float((full_logits - stream_logits).abs().max().item())
    assert len(set(mem_sizes)) == 1, f"Cache size grew! {mem_sizes}"
    return {"max_logit_diff": max_diff, "cache_bytes_b1": float(mem_sizes[0])}


# ==============================================================================
# 5. PRESETS & MAIN TRAINING LOOP
# ==============================================================================

PRESETS = {
    "smoke": dict(d_model=48,  d_key=24,  d_val=24,  n_layers=2, delay_lags=4, n_receptors=8),
    "1m":    dict(d_model=160, d_key=80,  d_val=80,  n_layers=4, delay_lags=4, n_receptors=12),
    "4m":    dict(d_model=288, d_key=144, d_val=144, n_layers=6, delay_lags=4, n_receptors=16),
    "10m":   dict(d_model=416, d_key=208, d_val=208, n_layers=8, delay_lags=4, n_receptors=16),
}


def run_training(args: argparse.Namespace) -> Dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    use_amp = (device.type == "cuda") and (args.dtype in ("bf16", "fp16"))
    amp_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    use_scaler = (device.type == "cuda") and (args.dtype == "fp16")
    scaler = (
        torch.amp.GradScaler("cuda", enabled=use_scaler)
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler")
        else torch.cuda.amp.GradScaler(enabled=use_scaler)
    )

    preset = PRESETS[args.preset]
    cfg = TriNetConfig(
        vocab_size=256,
        d_model=args.d_model or preset["d_model"],
        d_key=args.d_key or preset["d_key"],
        d_val=args.d_val or preset["d_val"],
        n_layers=args.n_layers or preset["n_layers"],
        delay_lags=args.delay_lags or preset["delay_lags"],
        n_receptors=args.n_receptors or preset["n_receptors"],
    )

    tokenizer = ByteTokenizer()
    trinet = TriNetForCausalLM(cfg).to(device)
    trinet_params = sum(p.numel() for p in trinet.parameters())

    baseline: Optional[MatchedTransformerLM] = None
    baseline_params = 0
    if not args.skip_baseline:
        baseline = build_matched_transformer(
            trinet_params, cfg, use_pos_emb=True, max_len=max(512, args.seq_len * 2)
        ).to(device)
        baseline_params = sum(p.numel() for p in baseline.parameters())

    print(f"Device: {device} | Preset: {args.preset} | AMP: {args.dtype if use_amp else 'fp32'}")
    print(f"TriNetForCausalLM:    {trinet_params:,} params")
    if baseline is not None:
        print(
            f"MatchedTransformerLM: {baseline_params:,} params "
            f"(ratio={baseline_params / trinet_params:.3f}x)"
        )

    train_stream = ByteStreamProvider(
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        text_column=args.text_column,
        local_text_path=args.local_text,
        smoke_test=args.smoke_test,
        seed=args.seed,
    )
    val_stream = ByteStreamProvider(
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        dataset_split=args.val_split,
        text_column=args.text_column,
        local_text_path=args.local_text,
        smoke_test=args.smoke_test,
        seed=args.seed + 999,
    )

    opt_tri = build_optimizer(trinet, lr=args.lr, weight_decay=args.weight_decay)
    opt_base = (
        build_optimizer(baseline, lr=args.lr, weight_decay=args.weight_decay) if baseline else None
    )

    os.makedirs(args.output_dir, exist_ok=True)
    history = []
    t0 = time.time()

    for step in range(1, args.steps + 1):
        lr_now = get_cosine_lr(step - 1, args.steps, args.warmup_steps, args.lr, args.min_lr)
        for pg in opt_tri.param_groups:
            pg["lr"] = lr_now
        if opt_base is not None:
            for pg in opt_base.param_groups:
                pg["lr"] = lr_now

        trinet.train()
        opt_tri.zero_grad(set_to_none=True)
        if baseline is not None and opt_base is not None:
            baseline.train()
            opt_base.zero_grad(set_to_none=True)

        tri_loss_acc = 0.0
        base_loss_acc = 0.0

        for _ in range(args.grad_accum):
            xb, yb = train_stream.sample_mixed_batch(
                args.batch_size, reasoning_mix=args.reasoning_mix
            )
            xb, yb = xb.to(device), yb.to(device)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out_tri = trinet(
                    input_ids=xb,
                    labels=yb,
                    use_cache=False,
                    metabolic_weight=args.metabolic_weight,
                )
                loss_tri = out_tri.loss / args.grad_accum

            scaler.scale(loss_tri).backward()
            tri_loss_acc += float(loss_tri.item())

            if baseline is not None:
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    loss_b, _ = baseline(input_ids=xb, labels=yb)
                    loss_b = loss_b / args.grad_accum
                scaler.scale(loss_b).backward()
                base_loss_acc += float(loss_b.item())

        scaler.unscale_(opt_tri)
        torch.nn.utils.clip_grad_norm_(trinet.parameters(), args.grad_clip)
        scaler.step(opt_tri)

        if baseline is not None and opt_base is not None:
            scaler.unscale_(opt_base)
            torch.nn.utils.clip_grad_norm_(baseline.parameters(), args.grad_clip)
            scaler.step(opt_base)

        scaler.update()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            elapsed = time.time() - t0
            history.append(
                {
                    "step": step,
                    "lr": lr_now,
                    "trinet_loss": tri_loss_acc,
                    "baseline_loss": base_loss_acc if baseline else None,
                    "elapsed_s": elapsed,
                }
            )
            base_str = f" | Transformer={base_loss_acc:.4f}" if baseline else ""
            print(
                f"Step {step:4d}/{args.steps} ({elapsed:5.1f}s) | "
                f"lr={lr_now:.2e} | TriNet={tri_loss_acc:.4f}{base_str}",
                flush=True,
            )

        if args.save_every > 0 and step % args.save_every == 0 and step < args.steps:
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
            trinet.save_pretrained(ckpt_dir, safe_serialization=True)
            tokenizer.save_metadata(ckpt_dir)

    # ---- Final Evaluation Suite ----
    eval_fillers = [6, 24] if args.smoke_test else [6, 30, 80]
    tri_prose = evaluate_prose_bpb(
        trinet, val_stream, args.batch_size, n_batches=4, device=device, is_trinet=True
    )
    tri_reason = evaluate_reasoning_suite(
        trinet, tokenizer, device, is_trinet=True, filler_lengths=eval_fillers
    )
    cache_check = verify_o1_cache(trinet, device)

    base_prose = None
    base_reason = None
    if baseline is not None:
        base_prose = evaluate_prose_bpb(
            baseline, val_stream, args.batch_size, n_batches=4, device=device, is_trinet=False
        )
        base_reason = evaluate_reasoning_suite(
            baseline, tokenizer, device, is_trinet=False, filler_lengths=eval_fillers
        )

    sample_prompt = "In a quiet workshop"
    p_ids = torch.tensor(
        [tokenizer.encode(sample_prompt, add_bos=True)], dtype=torch.long, device=device
    )
    trinet.eval()
    tri_gen_ids = trinet.generate(
        p_ids, max_new_tokens=32, do_sample=False, use_cache=True
    )[0].tolist()
    tri_gen_text = tokenizer.decode(tri_gen_ids)

    final_dir = os.path.join(args.output_dir, "trinet_hf")
    trinet.save_pretrained(final_dir, safe_serialization=True)
    tokenizer.save_metadata(final_dir)

    summary = {
        "trinet_params": trinet_params,
        "baseline_params": baseline_params,
        "trinet_prose": tri_prose,
        "baseline_prose": base_prose,
        "trinet_reasoning": tri_reason,
        "baseline_reasoning": base_reason,
        "o1_cache_check": cache_check,
        "sample_generation": tri_gen_text,
        "history": history,
    }
    with open(os.path.join(args.output_dir, "training_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== FINAL EVALUATION SUMMARY ===")
    print(f"TriNet Prose:      BPB={tri_prose['bpb']:.3f} | PPL={tri_prose['perplexity']:.2f}")
    if base_prose:
        print(f"Transformer Prose: BPB={base_prose['bpb']:.3f} | PPL={base_prose['perplexity']:.2f}")
    print(
        f"O(1) Cache Check:  max_logit_diff={cache_check['max_logit_diff']:.2e} | "
        f"bytes(B=1)={int(cache_check['cache_bytes_b1'])}"
    )
    for k, v in tri_reason.items():
        b_str = ""
        if base_reason:
            bv = base_reason[k]
            b_str = (
                f" || Transformer All={bv['all_acc']:5.1f}% "
                f"(Ovr={bv['overwrite_acc']:5.1f}%, 2H={bv['two_hop_acc']:5.1f}%)"
            )
        print(
            f"  {k:<10} | TriNet All={v['all_acc']:5.1f}% "
            f"(Ovr={v['overwrite_acc']:5.1f}%, 2H={v['two_hop_acc']:5.1f}%){b_str}"
        )
    print(f"Sample completion: {tri_gen_text!r}")

    if args.push_to_hub:
        trinet.push_to_hub(args.push_to_hub, safe_serialization=True)
        print(f"Pushed TriNetForCausalLM to Hugging Face Hub: {args.push_to_hub}")

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train byte-level TriNetForCausalLM + matched Transformer baseline"
    )
    p.add_argument("--preset", choices=list(PRESETS.keys()), default="1m")
    p.add_argument("--d-model", type=int, default=None)
    p.add_argument("--d-key", type=int, default=None)
    p.add_argument("--d-val", type=int, default=None)
    p.add_argument("--n-layers", type=int, default=None)
    p.add_argument("--delay-lags", type=int, default=None)
    p.add_argument("--n-receptors", type=int, default=None)

    p.add_argument(
        "--dataset", type=str, default=None, help="HF dataset ID, e.g. roneneldan/TinyStories"
    )
    p.add_argument("--dataset-config", type=str, default=None)
    p.add_argument("--dataset-split", type=str, default="train")
    p.add_argument("--val-split", type=str, default="validation")
    p.add_argument("--text-column", type=str, default="text")
    p.add_argument("--local-text", type=str, default=None, help="Local UTF-8 text file path")
    p.add_argument("--smoke-test", action="store_true", help="Fast offline smoke test")

    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--min-lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--metabolic-weight", type=float, default=0.005)
    p.add_argument("--reasoning-mix", type=float, default=0.25)

    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--output-dir", type=str, default="./runs/trinet_v01")
    p.add_argument("--skip-baseline", action="store_true")
    p.add_argument(
        "--push-to-hub", type=str, default=None, help="HF repo id, e.g. username/trinet-1m"
    )
    return p


if __name__ == "__main__":
    run_training(build_arg_parser().parse_args())
