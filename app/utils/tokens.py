from typing import Dict, List
from app import config

# Build token registry based on global config. No hardcoded IDs.
TOK: Dict[str, int] = dict(config.TOKENS)

# Dynamic numeric tokens for coordinates/sizes
NUM_TOKENS: Dict[int, int] = {i: TOK["BASE_NUM"] + i for i in range(max(config.DSL_ENUM_NUMS + [0]) + 1)}
# Dynamic color tokens
COLOR_TOKENS: Dict[int, int] = {c: TOK["BASE_COLOR"] + c for c in range(config.DSL_NUM_COLORS)}
# Rotations
ROT_TOKENS: Dict[int, int] = {k: TOK["BASE_ROT"] + k for k in config.DSL_ROTATIONS}
# Reflections
REFLECT_TOKENS: Dict[int, int] = {t: TOK["BASE_REFLECT"] + t for t in config.DSL_REFLECTIONS}

# Reverse maps
REV_NUM: Dict[int, int] = {v: k for k, v in NUM_TOKENS.items()}
REV_COLOR: Dict[int, int] = {v: k for k, v in COLOR_TOKENS.items()}
REV_ROT: Dict[int, int] = {v: k for k, v in ROT_TOKENS.items()}
REV_REFLECT: Dict[int, int] = {v: k for k, v in REFLECT_TOKENS.items()}

# Core vocabulary set
VOCAB: List[int] = sorted(set(TOK.values()) | set(NUM_TOKENS.values()) | set(COLOR_TOKENS.values()) | set(ROT_TOKENS.values()) | set(REFLECT_TOKENS.values()))
VOCAB_SIZE: int = max(VOCAB) + 1

# Arity map for ops
OP_ARITY: Dict[int, int] = {
    TOK["COMPOSE"]: 2,
    TOK["MAP_COLOR"]: 2,   # src_color, dst_color
    TOK["FILTER_EQ"]: 1,   # color
    TOK["ROTATE"]: 1,      # k
    TOK["REFLECT"]: 1,     # type
    TOK["CROP"]: 4,        # top, left, height, width
    TOK["PAD"]: 5,         # top, bottom, left, right, color
    # Extended ops
    TOK["FIND_COMPONENTS"]: 1,     # offset
    TOK["GET_BBOX"]: 1,            # color
    TOK["PAINT_OBJECT"]: 2,        # src_color, dst_color
    TOK["COUNT_COLOR"]: 1,         # color
    TOK["MAJORITY_COLOR"]: 1,      # tie_color
    TOK["TRANSLATE"]: 2,           # dy, dx
    TOK["DRAW_LINE"]: 5,           # y0, x0, y1, x1, color
    TOK["FILL_RECT"]: 5,           # top, left, height, width, color
    TOK["REPEAT_TILE"]: 2,         # ny, nx
    TOK["OVERLAY_UNION"]: 3,       # color_a, color_b, out_color
    TOK["OVERLAY_INTERSECT"]: 3,   # color_a, color_b, out_color
}

# Allowed op token ids (enable list in config controls availability)
ENABLED_OP_TOKENS = [TOK[name] for name in config.DSL_ENABLE_OPS if name in TOK]
