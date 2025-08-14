import os
import yaml
from typing import Any, Dict

# Module-level global configuration variables. They are populated by _load().
# Other modules should import from app.config, not hardcode literals.
CFG: Dict[str, Any] = {}

# Defaults (overridden by state/config.yaml)
DEFAULTS: Dict[str, Any] = {
    "dsl": {
        "max_depth": 4,
        "max_arity": 2,
        "num_colors": 10,
        "max_grid_size": 30,
        "enum_numeric_values": [0, 1, 2, 3, 4],
        "rotations": [0, 1, 2, 3],
        "reflections": [0, 1, 2, 3, 4],
        "enable_ops": [
            "MAP_COLOR", "FILTER_EQ", "ROTATE", "REFLECT", "CROP", "PAD", "COMPOSE"
        ],
    },
    "tokens": {
        "START": 1,
        "END": 2,
        "COMPOSE": 3,
        "MAP_COLOR": 10,
        "FILTER_EQ": 11,
        "ROTATE": 12,
        "REFLECT": 13,
        "CROP": 14,
        "PAD": 15,
        # Extended ops (IDs configurable via state/config.yaml)
        "FIND_COMPONENTS": 16,
        "GET_BBOX": 17,
        "PAINT_OBJECT": 18,
        "COUNT_COLOR": 19,
        "MAJORITY_COLOR": 20,
        "TRANSLATE": 21,
        "DRAW_LINE": 22,
        "FILL_RECT": 23,
        "REPEAT_TILE": 24,
        "OVERLAY_UNION": 25,
        "OVERLAY_INTERSECT": 26,
        "BASE_NUM": 100,
        "BASE_COLOR": 200,
        "BASE_ROT": 300,
        "BASE_REFLECT": 320,
    },
    "search": {
        "beam_width": 16,
        "max_nodes": 1000,
        "max_expansions_per_node": 32,
        "cache_results": True,
        "bad_score": -1e9,
        # Typed incremental search weights
        "typed_w_logprob": 1.0,
        "typed_w_heur": 1.0,
        # Numerical stability for entropy calculation
        "entropy_eps": 1e-12,
        "entropy_clip_max": 1.0,
        # RCR-inspired routing controls for heuristic scoring
        # If enabled, only a subset of I/O pairs will be used for heuristic scoring
        # to respect a context/token budget analogue. Strategy can be "entropy_size".
        "pair_subset_enable": True,
        "pair_subset_k": 2,
        "pair_subset_strategy": "entropy_size",
        # Optional policy-specific search limits (fallback to beam_width/max_nodes if unset)
        "policy_beam_width": 16,
        "policy_max_nodes": 1000,
    },
    "training": {
        "pseudo_max_depth": 3,
        "task_limit": 20,
        "batch_size": 64,
        "epochs": 5,
        "lr": 5e-4,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "entropy_coef": 0.0,
        "kl_coef": 0.0,
        "seed": 42,
    },
    "model": {
        "d_model": 128,
        "nhead": 4,
        "nlayers": 2,
        "dropout": 0.1,
        "max_tokens": 128,
        # CNN backbone knobs
        "use_cnn_backbone": True,
        "cnn_channels": [64, 128],  # per stage output channels
        "cnn_blocks_per_stage": [1, 1],
        "cnn_kernel_size": 3,
        "cnn_norm": "batch",  # "batch" or "layer" or "none"
        "cnn_pool": "avg",    # global pooling: "avg" or "max"
        # Input encoding
        "input_onehot": True,  # use one-hot over colors as input channels
        # Object-centric branch knobs
        "use_object_branch": True,
        "components_connectivity": 4,
        "components_background_color": 0,
        "obj_max_components": 64,
        "obj_hidden": 128,
        "obj_heads": 4,
        "obj_layers": 2,
        "obj_pool": "attn",   # "attn" or "mean"
        # Context fusion
        "context_fusion": "concat_proj",  # how to fuse grid+objects -> D
    },
    "inference": {
        "time_budget_s": 10,
        "beam_width": 16,
        "max_nodes": 1000,
        "use_policy": True,
        "num_workers": 1,
        # Disable time budget when value <= this threshold
        "time_budget_disable_threshold": 0.0,
        # Parallelization conditions
        "parallel_min_workers": 2,
        "parallel_min_tasks": 2,
        # Progress logging interval (tasks)
        "progress_interval": 1,
    },
}

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "state", "config.yaml")


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def reload_config() -> None:
    global CFG
    cfg = DEFAULTS.copy()
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            file_cfg = yaml.safe_load(f) or {}
        cfg = _deep_update(cfg, file_cfg)
    CFG = cfg


# Load once on import
reload_config()

# Convenience module-level globals to avoid hardcoded literals
DSL_MAX_DEPTH = CFG["dsl"]["max_depth"]
DSL_MAX_ARITY = CFG["dsl"]["max_arity"]
DSL_NUM_COLORS = CFG["dsl"]["num_colors"]
DSL_MAX_GRID_SIZE = CFG["dsl"]["max_grid_size"]
DSL_ENUM_NUMS = CFG["dsl"]["enum_numeric_values"]
DSL_ROTATIONS = CFG["dsl"]["rotations"]
DSL_REFLECTIONS = CFG["dsl"]["reflections"]
DSL_ENABLE_OPS = CFG["dsl"]["enable_ops"]

TOKENS = CFG["tokens"]
SEARCH_CFG = CFG["search"]
TRAIN_CFG = CFG["training"]
MODEL_CFG = CFG["model"]
INFER_CFG = CFG["inference"]

# Convenience constants from sections
SEARCH_BAD_SCORE = float(SEARCH_CFG.get("bad_score", -1e9))
