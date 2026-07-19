"""Per-layer, per-iteration metric recording — the step_callback passed to
FisherAdapTuneTrainer.

Four metrics per tracked layer, all read/computed at the same instant so
their curves share exactly the same x-axis (step) values:

  1. Fisher drift (JS-distance) -- the metric the paper's freeze mechanism
     is built on. Read from trainer._chunk_js_history (already tracked by
     the trainer for its own internal plot, just never aggregated per
     layer nor exposed outside it).
  2. Fisher magnitude -- read from trainer._chunk_fisher_ema, which the
     trainer overwrites in place every call and never keeps a history of
     anywhere else. Without it, a flat JS-drift curve is ambiguous: high
     but stable curvature (a good freeze candidate) looks identical to a
     layer that never had meaningful Fisher information to begin with.
  3. Gradient L2 norm -- classic optimization diagnostic, independent of
     any Fisher computation. Valid at the point this callback is invoked
     (see trainer.py) because it runs right after backward(), before
     optimizer.step().
  4. Relative weight update ||delta w|| / ||w|| -- what AdamW actually did
     to the weights, as opposed to the raw gradient signal before the
     optimizer step.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from src.utils.layers import resolve_layer_chunk_keys

_EPS = 1e-12


class LayerMetricRecorder:
    def __init__(
        self,
        model: nn.Module,
        layer_registry: Dict[str, str],
        chunk_specs_by_param,
    ):
        self.layer_registry = dict(layer_registry)
        self._chunk_keys_by_layer: Dict[str, List[str]] = {
            label: resolve_layer_chunk_keys(path, chunk_specs_by_param)
            for label, path in self.layer_registry.items()
        }
        for label, keys in self._chunk_keys_by_layer.items():
            if not keys:
                raise ValueError(
                    f"Layer '{label}' ({self.layer_registry[label]}) resolved to zero "
                    "Fisher chunks -- check the module path / naming convention."
                )

        self.history: Dict[str, Dict[str, List[Tuple[int, float]]]] = {
            "fisher_drift": {label: [] for label in self.layer_registry},
            "fisher_magnitude": {label: [] for label in self.layer_registry},
            "grad_norm": {label: [] for label in self.layer_registry},
            "relative_update": {label: [] for label in self.layer_registry},
        }
        self._prev_weights: Dict[str, torch.Tensor] = {}

    def __call__(self, step: int, model: nn.Module, trainer) -> None:
        for label, layer_path in self.layer_registry.items():
            chunk_keys = self._chunk_keys_by_layer[label]

            js_values = [
                trainer._chunk_js_history[k][-1][1]
                for k in chunk_keys
                if trainer._chunk_js_history.get(k)
            ]
            fisher_drift = float(sum(js_values) / len(js_values)) if js_values else 0.0

            mag_values = [
                float(trainer._chunk_fisher_ema[k].mean())
                for k in chunk_keys
                if k in trainer._chunk_fisher_ema
            ]
            fisher_magnitude = float(sum(mag_values) / len(mag_values)) if mag_values else 0.0

            module = model.get_submodule(layer_path)
            grad_sq_sum = torch.zeros((), device=next(module.parameters()).device)
            for p in module.parameters(recurse=False):
                if p.grad is not None:
                    grad_sq_sum = grad_sq_sum + p.grad.pow(2).sum()
            grad_norm = float(grad_sq_sum.sqrt())

            weight = torch.cat(
                [p.detach().reshape(-1) for p in module.parameters(recurse=False)]
            )
            prev_weight = self._prev_weights.get(label)
            if prev_weight is None:
                relative_update = 0.0
            else:
                delta_norm = float((weight - prev_weight).norm())
                w_norm = float(prev_weight.norm())
                relative_update = delta_norm / (w_norm + _EPS)
            self._prev_weights[label] = weight.clone()

            self.history["fisher_drift"][label].append((step, fisher_drift))
            self.history["fisher_magnitude"][label].append((step, fisher_magnitude))
            self.history["grad_norm"][label].append((step, grad_norm))
            self.history["relative_update"][label].append((step, relative_update))

    def as_dict(self) -> Dict[str, Dict[str, List[Tuple[int, float]]]]:
        return self.history
