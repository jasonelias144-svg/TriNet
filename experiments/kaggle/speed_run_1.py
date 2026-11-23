"""TriNet speed test on a Kaggle GPU (written by Claude, the TriNet
technical lead, for STATUS.md, Next 3).

Times one training step (forward, backward, Adam) of TriNetForCausalLM at
the sizes of train_trinet.py's presets and of Stage A, for two text
lengths and three batch sizes, on the GPU; and, for comparison, a small
Transformer (experiments/answers_only.py's, made larger). Records the
characters per second, the peak GPU memory, and which settings run out
of memory. Prints one JSON line per setting and writes speed.json.
"""
import json
import subprocess
import sys
import time

REPO = "https://github.com/jasonelias144-svg/TriNet"
COMMIT = "ef9af1f8640aa04b62dc0b63ce2e138b5b31654d"  # main after #8
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.46.3"], check=True)
subprocess.run(["git", "clone", "-q", REPO, "/kaggle/working/TriNet"], check=True)
subprocess.run(["git", "-C", "/kaggle/working/TriNet", "checkout", "-q", COMMIT], check=True)
sys.path.insert(0, "/kaggle/working/TriNet")
sys.path.insert(0, "/kaggle/working/TriNet/experiments")

import torch  # noqa: E402
import transformers  # noqa: E402
from trinet.configuration_trinet import TriNetConfig  # noqa: E402
from trinet.modeling_trinet import TriNetForCausalLM  # noqa: E402
from answers_only import SmallTransformer  # noqa: E402

dev = torch.device("cuda")
info = {"torch": torch.__version__, "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0), "gpus": torch.cuda.device_count(), "commit": COMMIT}
print(json.dumps(info), flush=True)

SIZES = {  # train_trinet.py PRESETS, and answers_only.SMALL (Stage A)
    "stage-a": dict(d_model=32, d_key=20, d_val=20, n_layers=2, delay_lags=4, n_receptors=8),
    "1m": dict(d_model=160, d_key=80, d_val=80, n_layers=4, delay_lags=4, n_receptors=12),
    "4m": dict(d_model=288, d_key=144, d_val=144, n_layers=6, delay_lags=4, n_receptors=16),
    "10m": dict(d_model=416, d_key=208, d_val=208, n_layers=8, delay_lags=4, n_receptors=16),
}
TRANSFORMERS = {  # (width, layers, feed-forward): about the sizes above
    "tf-1m": (128, 4, 512),
    "tf-10m": (384, 6, 1536),
}


def timed(model, batch, length, reps=4):
    torch.manual_seed(0)
    x = torch.randint(1, 256, (batch, length), device=dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    is_trinet = isinstance(model, TriNetForCausalLM)

    def step():
        if is_trinet:
            logits = model(input_ids=x, use_cache=False).logits
        else:
            logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    step()  # warm up
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    for _ in range(reps):
        step()
    torch.cuda.synchronize()
    sec = (time.time() - start) / reps
    return sec, torch.cuda.max_memory_allocated() / 2**30


results = [info]
for name, size in [*SIZES.items(), *TRANSFORMERS.items()]:
    for length in (128, 512):
        for batch in (8, 32, 128):
            row = {"model": name, "length": length, "batch": batch}
            try:
                torch.cuda.empty_cache()
                if name in SIZES:
                    model = TriNetForCausalLM(TriNetConfig(vocab_size=256, **size)).to(dev).train()
                else:
                    d, layers, ff = size
                    model = SmallTransformer(256, d=d, layers=layers, heads=4, ff=ff, max_len=1024).to(dev).train()
                row["parameters"] = sum(p.numel() for p in model.parameters())
                sec, mem = timed(model, batch, length)
                row.update(seconds_per_step=round(sec, 4), characters_per_second=round(batch * length / sec),
                           peak_gib=round(mem, 2))
            except torch.cuda.OutOfMemoryError:
                row["out_of_memory"] = True
            except Exception as e:  # recorded, not hidden
                row["error"] = f"{type(e).__name__}: {e}"[:300]
            finally:
                model = None
            print(json.dumps(row), flush=True)
            results.append(row)

with open("/kaggle/working/speed.json", "w") as f:
    json.dump(results, f, indent=1)
print("done", flush=True)
