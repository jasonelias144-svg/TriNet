---
license: cc0-1.0
language:
  - en
library_name: transformers
pipeline_tag: text-generation
tags:
  - trinet
  - custom_code
  - fast-weights
  - bio-inspired
  - byte-level
  - o1-memory
  - causal-lm
---

# TriNet — A 6-Archetype Heterogeneous Neural Language Model

**TriNet** replaces the uniform scalar-summation neuron with a small library of six specialized computational cell archetypes communicating over three functional channels (**Red**: excitatory drive, **Blue**: divisive shunting inhibition, **Yellow**: slow neuromodulatory attention/salience) and **dissolving fast synaptic weights**.

Instead of maintaining a growing $O(T)$ attention KV cache, each `TriNetLayer` maintains a constant-size $O(1)$ episodic working-memory state (`TriNetCache`) that binds novel associations in one shot, overwrites stale bindings via a local Delta rule, and dissolves unattended traces back toward baseline weights.

## Architecture: The Six Cell Archetypes

| Archetype | Core Mechanism | Role Inside `TriNetLayer` |
| :--- | :--- | :--- |
| **`TriChannelUnit`** | $W_{\text{base}} + F_{\text{fast}}(t)$ | Separates slow learned structure from temporary working memory |
| **`PyramidalUnit`** | Basal value vs. apical prediction error $v_t - k_t F_{\text{fast}}$ | Detects expectation mismatch and emits Yellow surprise |
| **`ShuntingInterneuron`** | Divisive normalization $R / (g_0 + B)$ | Scale-invariant local gain control without external LayerNorm |
| **`ModulatorNucleus`** | Slow tonic Yellow + sparse dead-zone gates $(\eta_t, \lambda_t)$ | Controls what writes into $F_{\text{fast}}$ and how fast traces dissolve |
| **`ReceptorBroadcastUnit`** | Rational Michaelis–Menten 3-color chromaticity bus | Scale-invariant composite-hue routing into the modulator |
| **`DelayLineUnit`** | Causal $K$-tap shift register (`Conv1d` + $O(1)$ ring buffer) | Parses local byte/character syntax and exact temporal intervals |

## Quickstart

`TriNetForCausalLM` registers with `AutoModelForCausalLM` and exports weights in `safetensors` format.

```python
import torch
from transformers import AutoModelForCausalLM

repo_id = "your-username/trinet-1m-v01"  # or local path to saved checkpoint
model = AutoModelForCausalLM.from_pretrained(repo_id, trust_remote_code=True).eval()

prompt = "In a quiet workshop"
input_ids = torch.tensor([[2] + list(prompt.encode("utf-8"))], dtype=torch.long)

with torch.no_grad():
    out_ids = model.generate(
        input_ids,
        max_new_tokens=64,
        do_sample=False,
        use_cache=True,  # uses O(1) constant-memory TriNetCache
    )[0].tolist()

decoded = bytes(i & 0xFF for i in out_ids if i not in (0, 2, 3)).decode("utf-8", errors="replace")
print(decoded)

Available Model Presets (train_trinet.py)
| Preset | d_model | d_key / d_val | Layers | Params (TriNetForCausalLM) | Matched Transformer |
|---|---|---|---|---|---|
| smoke | 48 | 24 / 24 | 2 | 59,708 | 71,648 |
| 1m | 160 | 80 / 80 | 4 | 837,512 | 836,320 |
| 4m | 288 | 144 / 144 | 6 | 3,797,460 | 3,797,280 |
| 10m | 416 | 208 / 208 | 8 | 10,330,288 | 10,312,832 |
Controlled Architectural Benchmarks (Random Init, Synthetic Suites)
Before corpus pre-training, the core 6-archetype architecture was evaluated from 100% random initialization against standard GRUs and Causal Transformers on two controlled stress tests:
 * Dynamic 3-Color Binding & Overwrite (928 params, trained at scale=1.0, d ∈ [1..8]):
   * Held 100.0% overall and overwrite accuracy zero-shot across 400× intensity shifts (0.05×–20×) and 200 noise distractors (T = 209) once the sparse dead-zone Yellow gate was enabled.
 * Raw Character-Level Direct / Overwrite / 2-Hop Reasoning (18k params, 2 layers, trained at T ≈ 45–61):
   * Reached 100.0% across Direct, Overwrite, and 2-Hop queries at T = 52, T = 100, and T = 200 (12× longer filler) in fixed O(1) state, outperforming equal-step 2-layer GRUs (63.7% at T = 200) and Causal Transformers (57.8–59.2% at T = 200).
(Insert real-corpus validation BPB, perplexity, and training_metrics.json results from your v0.1 run here before publishing a trained checkpoint.)
Scope and Honest Tradeoffs
 * Where TriNet excels: Constant O(1) runtime memory regardless of sequence length, rapid one-shot in-context variable binding and overwriting without modifying permanent weights, and scale-invariant composite-channel routing.
 * Where standard Transformers still win: Exact verbatim recall of arbitrary, unattended tokens far back in a long prompt. Because TriNetCache compresses working memory into fixed-size matrices (d_key × d_val per layer), unattended noise dissolves by design rather than persisting in an ever-growing KV cache.
 * Companion Auditable Implementation: A zero-float, fixed-point (SCALE = 1000) Rust implementation of all six archetypes and a cryptographically signed agent mind adapter (src/neuro.rs, src/neuro_mind.rs) accompanies this architecture in the open-worlds project for deterministic offline replay verification.
License
Dedicated to the public domain under CC0 1.0 Universal (CC0-1.0).
