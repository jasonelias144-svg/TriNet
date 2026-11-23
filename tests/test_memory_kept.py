"""Ours. What TriNet keeps for its backward pass (experiments/README.md,
"Speed on a GPU"): three batch x d_key x d_val matrices per layer per
character, the memory, the outer product written into it and the clamp's
input; most of what is kept."""
import torch

from trinet.configuration_trinet import TriNetConfig
from trinet.modeling_trinet import TriNetForCausalLM


def test_training_keeps_three_memory_matrices_per_layer_per_character():
    torch.manual_seed(0)
    cfg = TriNetConfig(vocab_size=256, d_model=48, d_key=24, d_val=20, n_layers=2, n_receptors=8)
    model = TriNetForCausalLM(cfg).train()
    batch, length = 3, 16
    params = {p.untyped_storage().data_ptr() for p in model.parameters()}
    matrices, other = {}, {}

    def pack(t):
        ptr, size = t.untyped_storage().data_ptr(), t.untyped_storage().nbytes()
        if ptr not in params:
            (matrices if size == batch * cfg.d_key * cfg.d_val * 4 else other)[ptr] = size
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        model(torch.randint(4, 256, (batch, length)), use_cache=False)
    assert len(matrices) == 3 * length * cfg.n_layers
    assert sum(matrices.values()) > sum(other.values())
