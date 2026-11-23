"""Ours. TriNet with one of its parts taken out, for the fair comparison of
STATUS.md, Next 1: what each part adds.

Each function changes a TriNetForCausalLM in place, after it is built and
before training, so that one part does nothing and cannot learn to, and
leaves everything else as it was (the same weights, the same random
numbers drawn). TriNetLayer's own code is not changed.

    no_bus(model)        the receptor bus gives a learned constant instead
                         of reading the signal (its hue receptors unused)
    no_surprise(model)   the yellow state and gates no longer read the
                         memory's surprise: they read only the bus and
                         their own slow state, which come from the input,
                         so the memory's write and retention no longer
                         depend on the memory itself
    fixed_gates(model)   the write and retain gates are learned constants
    no_shunt(model)      the blue shunt is off: the read is divided only
                         by the constant g0
    no_fast(model)       the write gate is shut: the fast memory stays at
                         zero, and the read is through W_base alone

The delay line is taken out by building the model with delay_lags=1
(fair_comparison.py), since its size is set when it is built.
"""
import torch
import torch.nn as nn


class ConstantBus(nn.Module):
    """In place of ReceptorBroadcastBus: the same number of outputs, each a
    learned constant."""

    def __init__(self, n_receptors: int):
        super().__init__()
        self.value = nn.Parameter(torch.full((n_receptors,), 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value.expand(*x.shape[:-1], self.value.shape[0])


def _freeze_columns(linear: nn.Linear, cols, value: float = 0.0):
    """Set some input columns of a linear layer's weight, and keep them so:
    their gradient is zeroed, so Adam never moves them."""
    with torch.no_grad():
        linear.weight[:, cols] = value
    mask = torch.ones_like(linear.weight)
    mask[:, cols] = 0.0
    linear.weight.register_hook(lambda g: g * mask)


def _freeze(param: nn.Parameter, value: float):
    with torch.no_grad():
        param.fill_(value)
    param.requires_grad_(False)


def no_bus(model):
    for layer in model.model.layers:
        layer.receptor_bus = ConstantBus(layer.receptor_bus.affinity_logits.shape[0])
    return model


def no_surprise(model):
    for layer in model.model.layers:
        col = layer.mod_in.in_features - 1  # mod_in reads [bus, surprise]
        _freeze_columns(layer.mod_in, [col])
        col = layer.yellow_to_write.in_features - 2  # the gates read [bus, surprise, tonic]
        _freeze_columns(layer.yellow_to_write, [col])
        _freeze_columns(layer.yellow_to_retain, [col])
    return model


def fixed_gates(model):
    for layer in model.model.layers:
        for gate in (layer.yellow_to_write, layer.yellow_to_retain):
            _freeze(gate.weight, 0.0)  # the bias alone sets the gate
    return model


def no_shunt(model):
    for layer in model.model.layers:
        _freeze(layer.shunt_w, -1e4)  # softplus(-1e4) is exactly 0
    return model


def no_fast(model):
    for layer in model.model.layers:
        _freeze(layer.yellow_to_write.weight, 0.0)
        _freeze(layer.yellow_to_write.bias, -1.0)  # the write rate is clamped to 0
    return model


ABLATIONS = {
    "trinet-no-bus": no_bus,
    "trinet-no-surprise": no_surprise,
    "trinet-fixed-gates": fixed_gates,
    "trinet-no-shunt": no_shunt,
    "trinet-no-fast": no_fast,
}
