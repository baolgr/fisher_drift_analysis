"""Single parametrized training entrypoint for the FisherAdapTune benchmark
(ResNet50 and ViT-small share one script -- see plan for why). Dataset is
config-driven (cfg["dataset"], see src/data/dataset.py's DATASET_REGISTRY)
-- CIFAR-10, CIFAR-100, and Downsampled ImageNet (32x32) as of this writing.

    python src/train.py --model resnet50  --config src/configs/config_resnet50.yaml  --disable-freeze
    python src/train.py --model resnet50  --config src/configs/config_resnet50.yaml
    python src/train.py --model vit_small --config src/configs/config_vit_small.yaml --disable-freeze
    python src/train.py --model vit_small --config src/configs/config_vit_small.yaml
"""

import argparse
import contextlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict

import torch
import torch.nn as nn
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.dataset import DATASET_REGISTRY, build_dataloaders
from src.fisher.fisher_core import build_all_chunk_specs, index_param_to_module
from src.fisher.trainer import FisherAdapTuneTrainer
from src.fisher.utils import EarlyStopping
from src.models.resnet_cifar import build_resnet50_cifar
from src.models.vit_cifar import build_vit_small_cifar
from src.utils.layers import (
    RESNET50_CIFAR_LAYERS,
    VIT_SMALL_LAYERS,
    classify_depth_group,
    classify_param_type,
    resolve_chunk_layer_and_block,
    resolve_layer_chunk_keys,
)
from src.utils.logs import parse_freeze_steps, parse_val_accuracy_history
from src.utils.metrics import LayerMetricRecorder
from src.utils.plotting import (
    plot_all_metrics_grid,
    plot_chunk_drift_heatmap,
    plot_chunk_drift_trajectories_by_tier,
    plot_drift_violin_by_group,
    plot_metric_evolution,
)


@dataclass(frozen=True)
class ModelSpec:
    build: Callable[..., nn.Module]
    layer_registry: Dict[str, str]
    display_name: str  # runs/<dataset>/<display_name>/... folder name


# Adding a future model: write the builder + layer registry (as today), add one
# ModelSpec entry here -- nothing else needs to change (mirrors DATASET_REGISTRY
# in src/data/dataset.py).
_MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "resnet50": ModelSpec(build_resnet50_cifar, RESNET50_CIFAR_LAYERS, "ResNet50"),
    "vit_small": ModelSpec(build_vit_small_cifar, VIT_SMALL_LAYERS, "ViT_Small"),
}

# Sidebar group order for plot_chunk_drift_heatmap, per model -- 4 groups to
# match the paper's own Fig 7/8 sidebar shape (Neck/EarlyBlk/LateBlk/Decoder).
_DEPTH_GROUP_ORDER = {
    "resnet50": ["Stem", "Layer1-2 (early)", "Layer3-4 (late)", "FC / Head"],
    "vit_small": ["Patch Embed", "Blocks 0-2 (early)", "Blocks 3-5 (late)", "Head"],
}
_PARAM_TYPE_GROUP_ORDER = ["Norm", "Attn", "MLP-FC", "Other (Conv/FC)"]


def _plot_fisher_group_summaries(
    trainer: FisherAdapTuneTrainer, model_name: str, metrics_dir: Path
) -> None:
    """Appendix B.1-style plots (paper Figures 7/8: JS-drift heatmap + spread,
    tiered trajectories, drift-by-parameter-type violin), over the *full*
    chunk population the trainer tracks -- unlike the curated 7-layer
    line plots above, which are capped at 7 for legend readability
    (see layers.py's module docstring), these don't need a cap since the
    heatmap/violin forms stay legible with hundreds of rows.
    """
    chunk_layer, chunk_block = resolve_chunk_layer_and_block(trainer._chunk_specs_by_param)
    chunk_final_js = {k: v[-1][1] for k, v in trainer._chunk_js_history.items() if v}

    layer_block_values: Dict[str, Dict[int, float]] = {}
    for chunk_key, js in chunk_final_js.items():
        layer = chunk_layer.get(chunk_key)
        block_idx = chunk_block.get(chunk_key)
        if layer is None or block_idx is None:
            continue
        layer_block_values.setdefault(layer, {})[block_idx] = js
    layer_group = {layer: classify_depth_group(layer, model_name) for layer in layer_block_values}
    plot_chunk_drift_heatmap(
        layer_block_values,
        layer_group,
        metrics_dir / "chunk_drift_heatmap.png",
        num_blocks=trainer.fisher_slice_blocks,
        group_order=_DEPTH_GROUP_ORDER[model_name],
    )

    plot_chunk_drift_trajectories_by_tier(
        trainer._chunk_js_history, metrics_dir / "chunk_drift_trajectories_by_tier.png"
    )

    chunk_param_type = {
        k: classify_param_type(chunk_layer[k], trainer._chunk_to_class.get(k, ""))
        for k in chunk_final_js
        if k in chunk_layer
    }
    plot_drift_violin_by_group(
        chunk_final_js,
        chunk_param_type,
        metrics_dir / "chunk_drift_violin_by_param_type.png",
        group_order=_PARAM_TYPE_GROUP_ORDER,
        title="Drift by parameter type",
    )


def build_model(name: str, cfg: Dict) -> nn.Module:
    builder = _MODEL_REGISTRY[name].build
    dataset_key = cfg.get("dataset", "cifar10")
    num_classes = cfg.get("num_classes", DATASET_REGISTRY[dataset_key].num_classes)
    if name == "vit_small":
        vit_cfg = cfg.get("vit", {})
        return builder(
            num_classes=num_classes,
            patch_size=vit_cfg.get("patch_size", 4),
            embed_dim=vit_cfg.get("embed_dim", 192),
            depth=vit_cfg.get("depth", 6),
            num_heads=vit_cfg.get("num_heads", 3),
            mlp_ratio=vit_cfg.get("mlp_ratio", 4.0),
            drop=vit_cfg.get("drop", 0.1),
        )
    return builder(num_classes=num_classes)


def make_train_step(loss_fn, device):
    def train_step(model, batch):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        return loss_fn(logits, labels)

    return train_step


def make_val_step(loss_fn, device):
    @torch.inference_mode()
    def val_step(model, batch):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = loss_fn(logits, labels)
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
        return {"loss": float(loss.item()), "accuracy": float(accuracy.item())}

    return val_step


def _assert_registry_resolves(layer_registry, chunk_specs_by_param) -> None:
    for label, path in layer_registry.items():
        keys = resolve_layer_chunk_keys(path, chunk_specs_by_param)
        if not keys:
            raise RuntimeError(
                f"Layer registry entry '{label}' ({path}) resolved to zero Fisher chunks -- "
                "check the module naming convention before training."
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(_MODEL_REGISTRY.keys()))
    parser.add_argument("--config", required=True)
    parser.add_argument("--disable-freeze", action="store_true")
    parser.add_argument("--num-epochs", type=int, default=None)
    return parser.parse_args()


class _Tee:
    """Duplicates writes to multiple streams -- lets train.log capture the same
    stdout/stderr a terminal/nohup would see, without losing the live console
    output people already rely on when following a run with tail -f.
    """

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


def run_training(model_name: str, cfg: Dict, disable_freeze: bool = False) -> Dict:
    # Opened first (before any print/log line, including trainer/Fisher setup)
    # so train.log always lands beside summary.json/metrics_plots/ for this
    # run, instead of relying on someone manually redirecting nohup and
    # renaming the file afterward.
    # run_name is an optional config override (default: model_freeze/nofreeze) --
    # lets several freeze-config variants of the same model run to distinct
    # runs/<Dataset>/<Model>/<run_name>/ directories (needed so parallel SLURM
    # jobs don't collide on the same output_dir mid-training, and so
    # scripts/compare_runs.py's recursive runs/**/summary.json glob picks all
    # of them up directly). The dataset/model subfolders keep experiments
    # across datasets from mixing together as more accumulate.
    default_run_name = f"{model_name}_{'nofreeze' if disable_freeze else 'freeze'}"
    dataset_display = DATASET_REGISTRY[cfg.get("dataset", "cifar10")].display_name
    model_display = _MODEL_REGISTRY[model_name].display_name
    output_dir = (
        Path(cfg.get("output_dir", "."))
        / dataset_display
        / model_display
        / cfg.get("run_name", default_run_name)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "train.log", "w") as log_file, contextlib.redirect_stdout(
        _Tee(sys.stdout, log_file)
    ), contextlib.redirect_stderr(_Tee(sys.stderr, log_file)):
        return _run_training(model_name, cfg, disable_freeze, output_dir)


def _run_training(model_name: str, cfg: Dict, disable_freeze: bool, output_dir: Path) -> Dict:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    torch.manual_seed(int(cfg.get("seed", 0)))

    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    model = build_model(model_name, cfg).to(device)
    layer_registry = _MODEL_REGISTRY[model_name].layer_registry

    # Fail fast on a registry/naming mismatch instead of producing empty plots later.
    param_to_modinfo = index_param_to_module(model)
    chunk_specs_by_param, _ = build_all_chunk_specs(
        model,
        param_to_modinfo,
        slice_blocks=cfg.get("fisher_slice_blocks", 4),
        slice_mode=cfg.get("fisher_slice_mode", "row"),
    )
    _assert_registry_resolves(layer_registry, chunk_specs_by_param)

    recorder = LayerMetricRecorder(model, layer_registry, chunk_specs_by_param)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get("learning_rate", 3e-4), weight_decay=0.0)
    num_epochs = cfg.get("num_epochs", 15)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * steps_per_epoch)
    early_stopping = EarlyStopping(patience=cfg.get("early_stopping_patience", 5))

    loss_fn = nn.CrossEntropyLoss()
    freeze_interval = cfg.get("freeze_interval", 300)
    if disable_freeze:
        freeze_interval = num_epochs * steps_per_epoch + 1

    trainer = FisherAdapTuneTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        train_step_fn=make_train_step(loss_fn, device),
        val_loader=val_loader,
        val_step_fn=make_val_step(loss_fn, device),
        scheduler=scheduler,
        early_stopping=early_stopping,
        num_epochs=num_epochs,
        weight_decay=cfg.get("weight_decay", 5e-5),
        output_dir=str(output_dir),
        fisher_ema_interval=cfg.get("fisher_ema_interval", 50),
        freeze_interval=freeze_interval,
        fisher_slice_mode=cfg.get("fisher_slice_mode", "row"),
        fisher_slice_blocks=cfg.get("fisher_slice_blocks", 4),
        js_distance_mode=cfg.get("js_distance_mode", "log"),
        chunk_selection_metric=cfg.get("chunk_selection_metric", "total_variation"),
        js_variance_lambda=cfg.get("js_variance_lambda", 1.0),
        disable_fisher_plots=True,
        step_callback=recorder,
        device=device,
    )
    trainer.fit()

    # Test set evaluated exactly once, at the end, never during training.
    val_step = make_val_step(loss_fn, device)
    model.eval()
    test_metrics_sum = {"loss": 0.0, "accuracy": 0.0}
    n_batches = 0
    for batch in test_loader:
        metrics = val_step(model, batch)
        for k, v in metrics.items():
            test_metrics_sum[k] += v
        n_batches += 1
    test_metrics = {k: v / n_batches for k, v in test_metrics_sum.items()}

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = int(sum(m.sum().item() for m in trainer._param_train_masks.values()))

    metrics_dir = output_dir / "metrics_plots"
    all_metrics = recorder.as_dict()
    # Multi-freeze configs (freeze_interval lowered so the threshold trigger
    # fires more than once, added 2026-07-20 -- see slurm/README.md's
    # "Multi-freeze batch" section) mean freeze_interval alone no longer
    # identifies every event, and early stopping can cut a run short before
    # a later planned event ever fires -- read back the actual "[Freeze]
    # Step N: ..." lines from train.log instead of computing multiples of
    # freeze_interval from config.
    accuracy_history = parse_val_accuracy_history(output_dir / "train.log", steps_per_epoch)
    freeze_steps = parse_freeze_steps(output_dir / "train.log")
    for metric_name, metric_history in all_metrics.items():
        plot_metric_evolution(
            metric_history, metric_name, metrics_dir / f"{metric_name}.png",
            freeze_steps=freeze_steps,
            # Overlay directly on fisher_drift.png only (report §7): that's
            # the one figure §4 shows can look like a real behavior change
            # right at the freeze step when it's actually a mean-recomposition
            # artifact -- the accuracy curve alongside it lets a reader see
            # the freeze-shock for what it is instead.
            accuracy_history=accuracy_history if metric_name == "fisher_drift" else None,
        )
    plot_all_metrics_grid(all_metrics, metrics_dir / "all_metrics_grid.png")
    _plot_fisher_group_summaries(trainer, model_name, metrics_dir)

    # Raw per-step/per-chunk history, so a future plotting-code change can
    # regenerate every PNG above from this run without re-training --
    # metrics_plots/*.png alone don't carry the underlying series back out.
    with open(metrics_dir / "history.json", "w") as f:
        json.dump(
            {
                "model": model_name,
                "freeze_steps": freeze_steps,
                "accuracy_history": accuracy_history,
                "metrics": all_metrics,
                "chunk_js_history": trainer._chunk_js_history,
                "chunk_to_class": trainer._chunk_to_class,
            },
            f,
        )

    summary = {
        "model": model_name,
        "disable_freeze": disable_freeze,
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "trainable_params": trainable_params,
        "total_params": total_params,
        "config": cfg,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.num_epochs is not None:
        cfg["num_epochs"] = args.num_epochs

    summary = run_training(args.model, cfg, disable_freeze=args.disable_freeze)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
