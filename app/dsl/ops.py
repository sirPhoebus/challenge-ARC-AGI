from __future__ import annotations
from typing import Tuple
import numpy as np
from app import config

# IMPORTANT: All constants via globals from config; no hardcoded values.
NUM_COLORS = config.DSL_NUM_COLORS
MAX_SIZE = config.DSL_MAX_GRID_SIZE

# Reflection mapping derived from configured reflection list order.
_REFLECT_OPS = [
    (lambda g: g.copy()),      # none
    (lambda g: np.flipud(g)),  # vertical
    (lambda g: np.fliplr(g)),  # horizontal
    (lambda g: g.T.copy()),    # diag_main
    (lambda g: np.rot90(g.T, 2)),  # diag_anti
]
REFLECT_FUNCS = {t: _REFLECT_OPS[i] for i, t in enumerate(list(config.DSL_REFLECTIONS)) if i < len(_REFLECT_OPS)}


def map_color(grid: np.ndarray, src: int, dst: int) -> np.ndarray:
    out = grid.copy()
    out[grid == int(src)] = int(dst)
    return out


def filter_eq(grid: np.ndarray, color: int) -> np.ndarray:
    out = np.zeros_like(grid)
    out[grid == int(color)] = int(color)
    return out


def rotate(grid: np.ndarray, k: int) -> np.ndarray:
    # Use provided rotation steps k; tokens ensure k is consistent with config.
    return np.rot90(grid, int(k))


def reflect(grid: np.ndarray, t: int) -> np.ndarray:
    t = int(t)
    fn = REFLECT_FUNCS.get(t, lambda g: g.copy())
    return fn(grid)


def crop(grid: np.ndarray, top: int, left: int, height: int, width: int) -> np.ndarray:
    h, w = grid.shape
    t = max(0, int(top)); l = max(0, int(left))
    hh = max(0, int(height)); ww = max(0, int(width))
    t = min(t, h)
    l = min(l, w)
    b = min(t + hh, h)
    r = min(l + ww, w)
    return grid[t:b, l:r].copy()


def pad(grid: np.ndarray, top: int, bottom: int, left: int, right: int, color: int) -> np.ndarray:
    t = max(0, int(top)); b = max(0, int(bottom)); l = max(0, int(left)); r = max(0, int(right))
    c = int(color)
    new_h = min(MAX_SIZE, grid.shape[0] + t + b)
    new_w = min(MAX_SIZE, grid.shape[1] + l + r)
    out = np.full((new_h, new_w), c, dtype=grid.dtype)
    ih, iw = grid.shape
    y0 = min(t, new_h)
    x0 = min(l, new_w)
    y1 = min(y0 + ih, new_h)
    x1 = min(x0 + iw, new_w)
    out[y0:y1, x0:x1] = grid[: y1 - y0, : x1 - x0]
    return out
