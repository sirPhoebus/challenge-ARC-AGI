from __future__ import annotations
from typing import Tuple
import numpy as np
from app import config

# IMPORTANT: All constants via globals from config; no hardcoded values.
NUM_COLORS = config.DSL_NUM_COLORS
MAX_SIZE = config.DSL_MAX_GRID_SIZE
COMPONENTS_CONNECTIVITY = int(config.MODEL_CFG.get("components_connectivity", 4))
BACKGROUND_COLOR = int(config.MODEL_CFG.get("components_background_color", 0))

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


# ----------------------------
# New grid-to-grid ops (minimal deterministic forms)
# ----------------------------

def translate(grid: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift grid down by dy and right by dx; out-of-bounds filled with BACKGROUND_COLOR.
    Offsets are clamped to be non-negative; shifting only down/right for simplicity.
    """
    h, w = grid.shape
    dy = max(0, int(dy))
    dx = max(0, int(dx))
    out = np.full_like(grid, BACKGROUND_COLOR)
    y0 = min(dy, h)
    x0 = min(dx, w)
    y1 = h
    x1 = w
    sy0 = 0
    sx0 = 0
    out[y0:y1, x0:x1] = grid[sy0:(sy0 + (y1 - y0)), sx0:(sx0 + (x1 - x0))]
    return out


def draw_line(grid: np.ndarray, y0: int, x0: int, y1: int, x1: int, color: int) -> np.ndarray:
    """Draw a straight line between (y0,x0) and (y1,x1) inclusive with the given color.
    Coordinates are clamped to grid bounds.
    """
    h, w = grid.shape
    y0 = max(0, min(int(y0), h - 1))
    x0 = max(0, min(int(x0), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x1 = max(0, min(int(x1), w - 1))
    c = int(color)
    out = grid.copy()
    dy = abs(y1 - y0)
    dx = abs(x1 - x0)
    sy = 1 if y0 < y1 else -1
    sx = 1 if x0 < x1 else -1
    err = (dx - dy)
    y, x = y0, x0
    while True:
        out[y, x] = c
        if y == y1 and x == x1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
        if y < 0 or y >= h or x < 0 or x >= w:
            break
    return out


def fill_rect(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> np.ndarray:
    """Fill a rectangle with color, clamped to grid bounds."""
    h, w = grid.shape
    t = max(0, int(top)); l = max(0, int(left))
    hh = max(0, int(height)); ww = max(0, int(width))
    b = min(h, t + hh)
    r = min(w, l + ww)
    out = grid.copy()
    out[t:b, l:r] = int(color)
    return out


def repeat_tile(grid: np.ndarray, ny: int, nx: int) -> np.ndarray:
    """Tile the input grid ny by nx times; crop to MAX_SIZE limits."""
    ny = max(1, int(ny))
    nx = max(1, int(nx))
    tiled = np.tile(grid, (ny, nx))
    H = min(MAX_SIZE, tiled.shape[0])
    W = min(MAX_SIZE, tiled.shape[1])
    return tiled[:H, :W].copy()


def get_bbox(grid: np.ndarray, color: int) -> np.ndarray:
    """Crop to the tight bounding box of the given color. If absent, return input unchanged."""
    c = int(color)
    ys, xs = np.where(grid == c)
    if ys.size == 0:
        return grid.copy()
    t, b = int(ys.min()), int(ys.max())
    l, r = int(xs.min()), int(xs.max())
    return grid[t:b+1, l:r+1].copy()


def paint_object(grid: np.ndarray, src_color: int, dst_color: int) -> np.ndarray:
    """Paint all cells of src_color to dst_color (component-wise collapse to a single color)."""
    out = grid.copy()
    out[grid == int(src_color)] = int(dst_color)
    return out


def find_components(grid: np.ndarray, offset: int) -> np.ndarray:
    """Relabel each non-background component with a color index offset by 'offset'.
    Background remains BACKGROUND_COLOR. Colors wrap within NUM_COLORS.
    """
    labels, n = connected_components_nonzero(grid, COMPONENTS_CONNECTIVITY, BACKGROUND_COLOR)
    out = grid.copy()
    off = int(offset) % max(1, NUM_COLORS)
    for i in range(n):
        col = (off + 1 + i) % NUM_COLORS
        out[labels == i] = col
    out[labels < 0] = BACKGROUND_COLOR
    return out


def count_color(grid: np.ndarray, color: int) -> np.ndarray:
    """Count occurrences of color and fill grid with (count mod NUM_COLORS)."""
    c = int(color)
    cnt = int(np.sum(grid == c))
    out_col = cnt % max(1, NUM_COLORS)
    return np.full_like(grid, out_col)


def majority_color(grid: np.ndarray, tie_color: int) -> np.ndarray:
    """Fill grid with the most frequent color; on ties, use tie_color."""
    vals, counts = np.unique(grid, return_counts=True)
    if vals.size == 0:
        return grid.copy()
    # Clamp colors to 0..NUM_COLORS-1 when counting
    best_idx = int(np.argmax(counts))
    maj = int(vals[best_idx])
    # Tie-breaker: if multiple share the max, use provided tie_color
    if np.sum(counts == counts[best_idx]) > 1:
        maj = int(tie_color) % max(1, NUM_COLORS)
    return np.full_like(grid, maj)


def overlay_union(grid: np.ndarray, color_a: int, color_b: int, out_color: int) -> np.ndarray:
    """Overlay union of masks for color_a and color_b, writing out_color where mask true."""
    a = int(color_a); b = int(color_b); oc = int(out_color)
    out = grid.copy()
    mask = (grid == a) | (grid == b)
    out[mask] = oc
    return out


def overlay_intersect(grid: np.ndarray, color_a: int, color_b: int, out_color: int) -> np.ndarray:
    """Overlay intersection region: cells of color_a adjacent (4-neigh) to any cell of color_b.
    Writes out_color at those a-cells.
    """
    a = int(color_a); b = int(color_b); oc = int(out_color)
    maskA = (grid == a)
    maskB = (grid == b)
    h, w = grid.shape
    adj = np.zeros_like(maskB)
    if h > 1:
        adj[:-1, :] |= maskB[1:, :]
        adj[1:, :] |= maskB[:-1, :]
    if w > 1:
        adj[:, :-1] |= maskB[:, 1:]
        adj[:, 1:] |= maskB[:, :-1]
    inter = maskA & adj
    out = grid.copy()
    out[inter] = oc
    return out


# ----------------------------
# Connected components (for features only)
# ----------------------------
def connected_components_nonzero(
    grid: np.ndarray,
    connectivity: int = COMPONENTS_CONNECTIVITY,
    background: int = BACKGROUND_COLOR,
) -> Tuple[np.ndarray, int]:
    """Label 2D grid components of non-background cells.

    Args:
        grid: (H, W) int grid of colors.
        connectivity: 4 or 8, from global config.
        background: color id treated as background (typically 0).

    Returns:
        labels: (H, W) int32 with labels in 0..(n-1) for object cells, -1 for background.
        n: number of labeled components.
    """
    a = np.asarray(grid)
    H, W = a.shape
    labels = np.full((H, W), -1, dtype=np.int32)
    vis = np.zeros((H, W), dtype=np.uint8)
    # Neighborhood deltas
    if int(connectivity) == 8:
        nbrs = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
    else:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    cur = 0
    for y in range(H):
        for x in range(W):
            if vis[y, x]:
                continue
            vis[y, x] = 1
            if int(a[y, x]) == int(background):
                continue
            # BFS/stack flood-fill
            stack = [(y, x)]
            labels[y, x] = cur
            while stack:
                cy, cx = stack.pop()
                for dy, dx in nbrs:
                    ny, nx = cy + dy, cx + dx
                    if ny < 0 or ny >= H or nx < 0 or nx >= W:
                        continue
                    if vis[ny, nx]:
                        continue
                    vis[ny, nx] = 1
                    if int(a[ny, nx]) == int(background):
                        continue
                    labels[ny, nx] = cur
                    stack.append((ny, nx))
            cur += 1
    return labels, int(cur)
