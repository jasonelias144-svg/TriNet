"""Interactive Hugging Face Space for inspecting TriNet's 6-archetype internals.

Visualizes token-by-token:
  1. Yellow Modulator gates (write gate eta, retention gate lam, tonic Yellow)
     and Pyramidal basal-vs-apical surprise.
  2. Dissolving fast-weight matrix F_fast(t) and single-step delta dF(t),
     scrubbable across every prompt and generated token.
  3. Exact O(1) TriNetCache footprint vs. O(T) Transformer KV cache growth.

Loads a trained checkpoint from `TRINET_REPO_ID` (HF Hub or local directory),
or falls back cleanly to a lightweight demo instance when run standalone.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import gradio as gr
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trinet.configuration_trinet import TriNetConfig
from trinet.modeling_trinet import TriNetForCausalLM

DEFAULT_REPO = os.environ.get("TRINET_REPO_ID", "").strip()
_MODEL_CACHE: Dict[str, Tuple[TriNetForCausalLM, str]] = {}


def load_trinet(source: str) -> Tuple[TriNetForCausalLM, str]:
    """Load from HF Hub / local folder via AutoModelForCausalLM, or boot a demo model."""
    key = source.strip() or "__demo__"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    if key == "__demo__":
        torch.manual_seed(42)
        cfg = TriNetConfig(
            vocab_size=256,
            d_model=64,
            d_key=32,
            d_val=32,
            n_layers=3,
            delay_lags=4,
            n_receptors=8,
        )
        model = TriNetForCausalLM(cfg).eval()
        n_params = sum(p.numel() for p in model.parameters())
        status = (
            f"Demo Mode (untrained 3-layer TriNet, {n_params:,} params). "
            "Pass a trained local path or Hugging Face repo ID above to inspect trained weights."
        )
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(key, trust_remote_code=True).eval()
        n_params = sum(p.numel() for p in model.parameters())
        status = f"Loaded checkpoint `{key}` ({n_params:,} params, {model.config.n_layers} layers)."

    _MODEL_CACHE[key] = (model, status)
    return model, status


def fmt_tok(tid: int) -> str:
    if tid == 2:
        return "<BOS>"
    if tid == 3:
        return "<EOS>"
    if tid == 0:
        return "<PAD>"
    if tid == 10:
        return "\\n"
    if tid == 9:
        return "\\t"
    if tid == 32:
        return "SP"
    if 33 <= tid <= 126:
        return chr(tid)
    return f"0x{tid:02X}"


@torch.no_grad()
def trace_trinet_internals(
    model: TriNetForCausalLM,
    prompt: str,
    max_new_tokens: int = 16,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Run O(1) stateful generation and record exact per-layer gate and F_fast traces."""
    model.eval()
    device = next(model.parameters()).device
    prompt_bytes = list(prompt.encode("utf-8"))
    if not prompt_bytes:
        prompt_bytes = [ord(" ")]
    tokens = [model.config.bos_token_id] + prompt_bytes
    n_prompt = len(tokens)

    # 1. Generate continuation using O(1) TriNetCache
    ids_cur = torch.tensor([tokens], dtype=torch.long, device=device)
    out = model(input_ids=ids_cur, past_key_values=None, use_cache=True)
    past = out.past_key_values
    gen_tokens: List[int] = []

    for _ in range(int(max_new_tokens)):
        logits = out.logits[:, -1, :]
        if temperature > 1e-4:
            probs = F.softmax(logits / temperature, dim=-1)
            nxt = int(torch.multinomial(probs, num_samples=1).item())
        else:
            nxt = int(torch.argmax(logits, dim=-1).item())
        gen_tokens.append(nxt)
        if nxt == model.config.eos_token_id:
            break
        out = model(
            input_ids=torch.tensor([[nxt]], dtype=torch.long, device=device),
            past_key_values=past,
            use_cache=True,
        )
        past = out.past_key_values

    full_ids = tokens + gen_tokens
    full_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
    seq_len = len(full_ids)
    labels = [fmt_tok(t) for t in full_ids]

    # 2. Exact glass-box forward trace across every layer
    h = model.model.embed_tokens(full_tensor)
    layer_traces = []

    for layer in model.model.layers:
        h_t = h.transpose(1, 2)
        bl = layer.delay_lags - 1
        if bl > 0:
            db = torch.zeros(1, layer.d_model, bl, device=device, dtype=h.dtype)
            h_cat = torch.cat([db, h_t], dim=-1)
        else:
            h_cat = h_t

        d_ctx = F.silu(layer.delay_conv(h_cat).transpose(1, 2))
        rec_rby = F.relu(layer.receptor_bus.to_rby(d_ctx))[0].cpu().numpy()
        rec_feat = layer.receptor_bus(d_ctx)

        k_write = F.silu(layer.W_key(d_ctx))
        k_write = k_write / torch.sqrt((k_write * k_write).sum(dim=-1, keepdim=True) + 1e-4)
        q_read = F.silu(layer.W_query(d_ctx))
        q_read = q_read / torch.sqrt((q_read * q_read).sum(dim=-1, keepdim=True) + 1e-4)
        v_basal_all = layer.W_val(h)

        f_fast = torch.zeros(1, layer.d_key, layer.d_val, device=device, dtype=h.dtype)
        y_tonic = torch.zeros(1, 1, device=device, dtype=h.dtype)
        td = torch.sigmoid(layer.tonic_decay_logit)
        g0 = torch.exp(layer.log_g0.clamp(-4.0, 1.0))
        sw = F.softplus(layer.shunt_w)
        rspan = 1.0 - layer.min_retention

        etas, lams, surprises, yellows, blue_shunts, f_norms = [], [], [], [], [], []
        f_snaps, df_snaps = [], []
        mem_outs = []

        for t in range(seq_len):
            qt = q_read[:, t, :]
            kt = k_write[:, t, :]
            vt = v_basal_all[:, t, :]
            rt = rec_feat[:, t, :]

            rr = torch.matmul(qt, layer.W_base) + torch.bmm(qt.unsqueeze(1), f_fast).squeeze(1)
            bs = (F.relu(rr) * sw).sum(dim=-1, keepdim=True)
            mem_outs.append(rr / (g0 + bs))

            va = torch.bmm(kt.unsqueeze(1), f_fast).squeeze(1)
            de = vt - va
            surp = torch.sqrt((de * de).mean(dim=-1, keepdim=True) + 1e-6)
            y_tonic = torch.clamp(
                td * y_tonic + F.relu(layer.mod_in(torch.cat([rt, surp], dim=-1))), 0.0, 2.0
            )

            gi = torch.cat([rt, surp, y_tonic], dim=-1)
            eta = layer.max_write_rate * torch.clamp(layer.yellow_to_write(gi), 0.0, 1.0)
            lam = layer.min_retention + rspan * torch.clamp(
                layer.yellow_to_retain(gi), 0.0, 1.0
            )

            prev_f = f_fast
            f_fast = torch.clamp(
                lam.unsqueeze(-1) * f_fast
                + eta.unsqueeze(-1) * torch.bmm(kt.unsqueeze(2), de.unsqueeze(1)),
                -layer.max_fast_weight,
                layer.max_fast_weight,
            )
            df = f_fast - prev_f

            etas.append(float(eta.item()))
            lams.append(float(lam.item()))
            surprises.append(float(surp.item()))
            yellows.append(float(y_tonic.item()))
            blue_shunts.append(float(bs.item()))
            f_norms.append(float(torch.norm(f_fast).item()))
            f_snaps.append(f_fast[0].cpu().numpy().copy())
            df_snaps.append(df[0].cpu().numpy().copy())

        fused = torch.cat([torch.stack(mem_outs, dim=1), d_ctx, rec_feat], dim=-1)
        h = layer.norm(h + layer.out_proj(fused))

        layer_traces.append(
            {
                "eta": np.array(etas),
                "lam": np.array(lams),
                "surprise": np.array(surprises),
                "yellow": np.array(yellows),
                "blue_shunt": np.array(blue_shunts),
                "f_norm": np.array(f_norms),
                "f_snaps": f_snaps,
                "df_snaps": df_snaps,
                "rby": rec_rby,
            }
        )

    specials = {model.config.pad_token_id, model.config.bos_token_id, model.config.eos_token_id}
    gen_text = bytes(i & 0xFF for i in gen_tokens if i not in specials).decode(
        "utf-8", errors="replace"
    )
    return {
        "config": model.config,
        "labels": labels,
        "n_prompt": n_prompt,
        "generated_text": gen_text,
        "layer_traces": layer_traces,
        "cache_bytes": past.memory_bytes(),
    }


def plot_modulator_timeline(trace: Dict[str, Any], layer_idx: int, selected_step: int):
    lt = trace["layer_traces"][layer_idx]
    labels = trace["labels"]
    seq_len = len(labels)
    xs = np.arange(seq_len)
    t_sel = int(np.clip(selected_step, 0, seq_len - 1))

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 6.2), sharex=True)
    fig.patch.set_facecolor("white")

    axes[0].plot(xs, lt["eta"], color="#C88A00", linewidth=2.0, marker="o", markersize=3.5, label="Write Gate eta (Yellow)")
    axes[0].plot(xs, lt["lam"], color="#1F4E79", linewidth=1.8, linestyle="--", label="Retention Gate lam")
    axes[0].axvline(t_sel, color="#111111", linestyle=":", linewidth=1.4)
    axes[0].axvline(trace["n_prompt"] - 0.5, color="#888888", linestyle="-.", linewidth=1.0, label="Prompt / Gen")
    axes[0].set_ylabel("Gate Value")
    axes[0].set_title(f"Layer {layer_idx} -- Yellow Modulator Gates, Pyramidal Surprise & Working-Memory Load")
    axes[0].legend(loc="upper right", fontsize=8, frameon=True)
    axes[0].grid(alpha=0.25)

    axes[1].fill_between(xs, 0, lt["surprise"], color="#B22222", alpha=0.25, label="Pyramidal Surprise ||v - k*F||")
    axes[1].plot(xs, lt["surprise"], color="#B22222", linewidth=1.6)
    axes[1].plot(xs, lt["yellow"], color="#D4A017", linewidth=1.8, label="Modulator Tonic Yellow")
    axes[1].axvline(t_sel, color="#111111", linestyle=":", linewidth=1.4)
    axes[1].set_ylabel("Surprise / Yellow")
    axes[1].legend(loc="upper right", fontsize=8, frameon=True)
    axes[1].grid(alpha=0.25)

    axes[2].plot(xs, lt["f_norm"], color="#111111", linewidth=2.0, label="Working Memory ||F_fast(t)||_F")
    axes[2].plot(xs, lt["blue_shunt"], color="#2E6B9E", linewidth=1.5, linestyle="--", label="Blue Shunting Pool B(t)")
    axes[2].axvline(t_sel, color="#111111", linestyle=":", linewidth=1.4)
    axes[2].set_ylabel("Norm / Shunt")
    axes[2].set_xticks(xs)
    axes[2].set_xticklabels(labels, fontsize=8, rotation=45 if seq_len > 28 else 0)
    axes[2].legend(loc="upper right", fontsize=8, frameon=True)
    axes[2].grid(alpha=0.25)

    plt.tight_layout()
    return fig


def plot_fast_weight_heatmaps(trace: Dict[str, Any], layer_idx: int, selected_step: int):
    lt = trace["layer_traces"][layer_idx]
    t_sel = int(np.clip(selected_step, 0, len(trace["labels"]) - 1))
    f_mat = lt["f_snaps"][t_sel]
    df_mat = lt["df_snaps"][t_sel]
    tok_lbl = trace["labels"][t_sel]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    vmax_f = max(0.05, float(np.max(np.abs(f_mat))))
    vmax_df = max(0.05, float(np.max(np.abs(df_mat))))

    im1 = ax1.imshow(f_mat, cmap="RdBu_r", vmin=-vmax_f, vmax=vmax_f, aspect="auto")
    ax1.set_title(f"F_fast(t={t_sel}, tok={tok_lbl!r}) | ||F||={lt['f_norm'][t_sel]:.3f}")
    ax1.set_xlabel("Value Dim (d_val)")
    ax1.set_ylabel("Key Dim (d_key)")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(df_mat, cmap="RdBu_r", vmin=-vmax_df, vmax=vmax_df, aspect="auto")
    ax2.set_title(f"dF(t={t_sel}) = F(t) - F(t-1) | eta={lt['eta'][t_sel]:.3f}, lam={lt['lam'][t_sel]:.3f}")
    ax2.set_xlabel("Value Dim (d_val)")
    ax2.set_ylabel("Key Dim (d_key)")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


def plot_o1_memory_comparison(config: TriNetConfig, seq_len: int):
    ts = np.arange(1, max(seq_len, 256) + 1)
    trinet_kb = np.full_like(
        ts,
        config.n_layers
        * (config.d_key * config.d_val + 1 + config.d_model * max(0, config.delay_lags - 1))
        * 4
        / 1024.0,
        dtype=np.float64,
    )
    kv_kb = (2 * config.n_layers * ts * config.d_model * 4) / 1024.0

    fig, ax = plt.subplots(figsize=(10.5, 3.0))
    ax.plot(ts, trinet_kb, color="#111111", linewidth=2.2, label="TriNetCache -- O(1) Constant State")
    ax.plot(ts, kv_kb, color="#B22222", linewidth=1.8, linestyle="--", label="Transformer KV Cache -- O(T) Linear Growth")
    ax.set_xlabel("Sequence Length T (tokens)")
    ax.set_ylabel("Runtime State (KB, B=1)")
    ax.set_title("Constant-Memory O(1) Fast-Weight State vs. Growing O(T) KV Cache")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    return fig


def run_and_plot(source: str, prompt: str, max_new_tokens: int, temperature: float, layer_idx: int):
    model, status = load_trinet(source)
    n_layers = model.config.n_layers
    l_idx = int(np.clip(layer_idx, 0, n_layers - 1))

    trace = trace_trinet_internals(
        model, prompt, max_new_tokens=int(max_new_tokens), temperature=float(temperature)
    )
    last_step = len(trace["labels"]) - 1

    fig_time = plot_modulator_timeline(trace, l_idx, last_step)
    fig_heat = plot_fast_weight_heatmaps(trace, l_idx, last_step)
    fig_mem = plot_o1_memory_comparison(trace["config"], seq_len=max(256, len(trace["labels"])))

    completion_md = (
        f"**Status:** {status}\n\n"
        f"**Prompt:** `{prompt}`  \n"
        f"**Generated ({len(trace['labels']) - trace['n_prompt']} bytes):** `{trace['generated_text']}`  \n"
        f"**Constant `TriNetCache` Footprint (B=1):** `{trace['cache_bytes']:,} bytes` across `{len(trace['labels'])}` tokens."
    )
    return (
        trace,
        completion_md,
        gr.update(maximum=n_layers - 1, value=l_idx),
        gr.update(maximum=last_step, value=last_step),
        fig_time,
        fig_heat,
        fig_mem,
    )


def scrub_view(trace: Dict[str, Any] | None, layer_idx: int, step_idx: int):
    if trace is None:
        return None, None
    l_idx = int(np.clip(layer_idx, 0, len(trace["layer_traces"]) - 1))
    s_idx = int(np.clip(step_idx, 0, len(trace["labels"]) - 1))
    return (
        plot_modulator_timeline(trace, l_idx, s_idx),
        plot_fast_weight_heatmaps(trace, l_idx, s_idx),
    )


with gr.Blocks(title="TriNet — 6-Archetype Fast-Weight Glass-Box Inspector") as demo:
    gr.Markdown(
        "# TriNet Glass-Box Inspector\n"
        "Inspect live **Red / Blue / Yellow** modulator dynamics, **Pyramidal surprise**, "
        "and **dissolving fast weights $F_{\\text{fast}}(t)$** token by token in $O(1)$ memory."
    )

    trace_state = gr.State(value=None)

    with gr.Row():
        source_box = gr.Textbox(
            label="Checkpoint Path or Hugging Face Repo ID (leave blank for Demo Mode)",
            value=DEFAULT_REPO,
            placeholder="e.g. your-username/trinet-1m-v01 or ./runs/trinet_v01/trinet_hf",
        )
    with gr.Row():
        prompt_box = gr.Textbox(
            label="Input Prompt (UTF-8 Byte Stream)",
            value="a=3;b=5;xyz.a=7;xyz?a>",
            lines=2,
        )
    with gr.Row():
        max_new_slider = gr.Slider(0, 64, value=12, step=1, label="Max New Bytes")
        temp_slider = gr.Slider(0.0, 1.5, value=0.0, step=0.05, label="Temperature (0 = Greedy)")
        run_btn = gr.Button("Run & Trace Internals", variant="primary")

    gr.Examples(
        examples=[
            ["a=3;b=5;xyz.a=7;xyz?a>"],
            ["c=0;g=1;e=4;zzz.d=c;g=5;fxzy?d>"],
            ["In a quiet workshop by the river, "],
        ],
        inputs=[prompt_box],
    )

    summary_md = gr.Markdown()

    with gr.Row():
        layer_slider = gr.Slider(0, 2, value=0, step=1, label="Inspect Layer")
        step_slider = gr.Slider(0, 32, value=0, step=1, label="Scrub Token Step (t)")

    timeline_plot = gr.Plot(label="Yellow Modulator Gates, Pyramidal Surprise & Memory Norm")
    heatmap_plot = gr.Plot(label="Dissolving Fast-Weight Matrix F_fast(t) & Step Update dF(t)")
    memory_plot = gr.Plot(label="O(1) TriNetCache vs. O(T) Transformer KV Cache")

    run_btn.click(
        fn=run_and_plot,
        inputs=[source_box, prompt_box, max_new_slider, temp_slider, layer_slider],
        outputs=[
            trace_state,
            summary_md,
            layer_slider,
            step_slider,
            timeline_plot,
            heatmap_plot,
            memory_plot,
        ],
    )
    layer_slider.change(
        fn=scrub_view,
        inputs=[trace_state, layer_slider, step_slider],
        outputs=[timeline_plot, heatmap_plot],
    )
    step_slider.change(
        fn=scrub_view,
        inputs=[trace_state, layer_slider, step_slider],
        outputs=[timeline_plot, heatmap_plot],
    )

if __name__ == "__main__":
    demo.launch()
