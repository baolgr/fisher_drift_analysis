"""Layer registries for the CIFAR-10 benchmark, and resolution from a
human-readable layer path to the internal Fisher chunk keys the trainer
tracks per-chunk state under.

For each ResNet50 Bottleneck block, `conv2` (the 3x3 spatial conv) is used
as the representative "convolution" for that stage rather than `conv1`/
`conv3` (1x1 channel projections, closer to a per-pixel Linear than a
convolution). `layer4.2.conv3` is kept as the one exception, to compare
directly against `fc` (both are essentially linear projections).

For the ViT, block indices 0/2/5 (out of depth=6) spread roughly evenly
across early/mid/late, and attn.qkv/mlp.fc1/mlp.fc2 are sampled at both
ends specifically to cross the two axes requested together: early-vs-late
and attention-vs-MLP.
"""

import re
from typing import Dict, List, Optional, Tuple

from src.fisher.fisher_core import ChunkSpec

RESNET50_CIFAR_LAYERS: Dict[str, str] = {
    "stem_conv": "conv1",
    "layer1.0.conv2": "layer1.0.conv2",
    "layer2.2.conv2": "layer2.2.conv2",
    "layer3.3.conv2": "layer3.3.conv2",
    "layer4.2.conv2": "layer4.2.conv2",
    "layer4.2.conv3": "layer4.2.conv3",
    "fc": "fc",
}

VIT_SMALL_LAYERS: Dict[str, str] = {
    "patch_embed.proj": "patch_embed.proj",
    "blocks.0.attn.qkv": "blocks.0.attn.qkv",
    "blocks.0.mlp.fc1": "blocks.0.mlp.fc1",
    "blocks.2.attn.qkv": "blocks.2.attn.qkv",
    "blocks.5.attn.qkv": "blocks.5.attn.qkv",
    "blocks.5.mlp.fc2": "blocks.5.mlp.fc2",
    "head": "head",
}


def resolve_layer_chunk_keys(
    layer_path: str,
    chunk_specs_by_param: Dict[str, List[ChunkSpec]],
) -> List[str]:
    """Resolve a layer's module path to the chunk keys the trainer tracks it under.

    Not an exact match: for attention QKV layers, fisher_core.py assigns
    chunk.layer_key = f"{layer_path}.q" / ".k" / ".v" (split by head),
    never the bare layer_path. An exact-equality implementation would
    silently return an empty list for every attention layer.
    """
    keys: List[str] = []
    prefix = layer_path + "."
    for specs in chunk_specs_by_param.values():
        for spec in specs:
            if spec.layer_key == layer_path or spec.layer_key.startswith(prefix):
                keys.append(spec.key)
    return keys


# ---------------------------------------------------------------------------
# Full-chunk-population grouping for the Appendix B.1 style plots (paper
# Figures 7/8: per-parameter-group JS-drift heatmap + within-layer spread +
# drift-by-component-type violin). Unlike the curated 7-layer registries
# above (built for a legend-per-line plot, capped at 7 for readability),
# these operate on *all* chunks the trainer tracks -- the paper's own
# figures make that scale legible precisely by dropping per-row labels in
# favor of a heatmap + a colored component sidebar, so there's no need to
# curate here.
# ---------------------------------------------------------------------------

_BLOCK_SUFFIX_RE = re.compile(r"(?:row|col)?block(\d+)$")


def resolve_chunk_layer_and_block(
    chunk_specs_by_param: Dict[str, List[ChunkSpec]],
) -> Tuple[Dict[str, str], Dict[str, Optional[int]]]:
    """chunk key -> (layer_key, column-block index), for the Fig 7(a)-style
    layer x column-block heatmap.

    The block index is parsed back out of the chunk key's ``rowblockN`` /
    ``colblockN`` / QKV ``blockN`` suffix (see
    ``_build_chunk_specs_for_param`` in fisher_core.py) rather than tracked
    separately, since ChunkSpec itself has no numeric block field -- only
    None for the ndim<2-under-column-slice-mode fallback (a single ``.full``
    chunk), which none of this repo's configs hit (both set
    ``fisher_slice_mode: row``).
    """
    chunk_layer: Dict[str, str] = {}
    chunk_block: Dict[str, Optional[int]] = {}
    for specs in chunk_specs_by_param.values():
        for spec in specs:
            chunk_layer[spec.key] = spec.layer_key
            match = _BLOCK_SUFFIX_RE.search(spec.key)
            chunk_block[spec.key] = int(match.group(1)) if match else None
    return chunk_layer, chunk_block


def classify_depth_group(layer_key: str, model_name: str) -> str:
    """4-way depth/component grouping for the heatmap's row sidebar, mirroring
    the paper's Fig 7 (Enc. Neck / Enc. Blk early / Enc. Blk late / Mask
    Decoder) and Fig 8 (Patch Embed / Enc Blk 1-2 / Enc Blk 3-4 / Decoder)
    sidebars -- same 4-group shape, relabeled for this benchmark's two models.
    """
    if model_name == "vit_small":
        if layer_key.startswith("patch_embed") or layer_key in ("cls_token", "pos_embed"):
            # cls_token/pos_embed are raw nn.Parameters (no owning Conv2d/Linear/
            # Norm submodule), so they're invisible to AdaFisherBackbone's hooks
            # and never appear in the curated VIT_SMALL_LAYERS registry (see
            # plan.md) -- but build_all_chunk_specs still builds ChunkSpecs for
            # every trainable parameter regardless of hook coverage, so they DO
            # reach this classifier. Grouped with Patch Embed: same "how the raw
            # input becomes the initial token sequence" stage.
            return "Patch Embed"
        if layer_key == "head":
            return "Head"
        if layer_key.startswith("blocks."):
            block_idx = int(layer_key.split(".")[1])
            return "Blocks 0-2 (early)" if block_idx <= 2 else "Blocks 3-5 (late)"
        if layer_key == "norm":
            return "Blocks 3-5 (late)"  # final post-blocks LayerNorm, normalizes block 5's output
        return "Other"
    if model_name == "resnet50":
        if layer_key.startswith("conv1") or layer_key.startswith("bn1"):
            return "Stem"
        if layer_key == "fc":
            return "FC / Head"
        if layer_key.startswith("layer1.") or layer_key.startswith("layer2."):
            return "Layer1-2 (early)"
        if layer_key.startswith("layer3.") or layer_key.startswith("layer4."):
            return "Layer3-4 (late)"
        return "Other"
    raise ValueError(f"Unknown model_name: {model_name!r}")


def classify_param_type(layer_key: str, cls_name: str) -> str:
    """Norm / Attn / MLP-FC / Other(Conv+FC) grouping for the violin plot,
    mirroring the paper's Fig 8(c) "Drift by Parameter Type" (Norm / Attn /
    MLP-FC). Attn and MLP-FC are both nn.Linear in this codebase (see
    src/models/vit_cifar.py), so cls_name alone can't split them -- only
    layer_key's ``.attn.``/``.mlp.`` path segment can, same reasoning as
    resolve_layer_chunk_keys's prefix-match above. ResNet50 has no
    attention/MLP axis, so its chunks only ever land in Norm or the
    catch-all -- an honest Norm-vs-everything-else split for that model,
    not a mislabel.
    """
    if cls_name in ("LayerNorm", "BatchNorm2d"):
        return "Norm"
    if ".attn." in layer_key:
        return "Attn"
    if ".mlp." in layer_key:
        return "MLP-FC"
    return "Other (Conv/FC)"
