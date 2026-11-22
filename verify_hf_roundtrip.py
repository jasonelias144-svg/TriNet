import tempfile
import torch
from transformers import AutoModelForCausalLM
from trinet.configuration_trinet import TriNetConfig
from trinet.modeling_trinet import TriNetForCausalLM

torch.manual_seed(42)
cfg = TriNetConfig(vocab_size=256, d_model=64, d_key=32, d_val=32, n_layers=3)
model = TriNetForCausalLM(cfg).eval()

with tempfile.TemporaryDirectory() as save_dir:
    model.save_pretrained(save_dir, safe_serialization=True)
    reloaded = AutoModelForCausalLM.from_pretrained(save_dir, trust_remote_code=True).eval()

    prompt = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        gen_orig = model.generate(prompt, max_new_tokens=24, do_sample=False, use_cache=True)
        gen_reloaded = reloaded.generate(prompt, max_new_tokens=24, do_sample=False, use_cache=True)
        gen_nocache = reloaded.generate(prompt, max_new_tokens=24, do_sample=False, use_cache=False)

        assert torch.equal(gen_orig, gen_reloaded), "Reloaded generation mismatch!"
        assert torch.equal(gen_reloaded, gen_nocache), "Cached vs uncached generation mismatch!"

        out_cached = reloaded(prompt, use_cache=True)
        assert out_cached.past_key_values.get_seq_length() == 8
        mem_before = out_cached.past_key_values.memory_bytes()
        out_next = reloaded(prompt[:, -1:], past_key_values=out_cached.past_key_values, use_cache=True)
        assert out_next.past_key_values.memory_bytes() == mem_before

    print("Real Hugging Face AutoModelForCausalLM + safetensors + O(1) generate() PASSED!")
