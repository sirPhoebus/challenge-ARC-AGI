from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
from app import config
from app.utils import tokens as T
from app.dsl import ops as OP

# Deterministic, vectorized NumPy interpreter of DSL token sequences.


def _tok_to_num(tok: int) -> int:
    if tok in T.REV_NUM:
        return int(T.REV_NUM[tok])
    if tok in T.REV_COLOR:
        return int(T.REV_COLOR[tok])
    if tok in T.REV_ROT:
        return int(T.REV_ROT[tok])
    if tok in T.REV_REFLECT:
        return int(T.REV_REFLECT[tok])
    return int(tok)


def _exec_ast(ast: Tuple, grid: np.ndarray) -> np.ndarray:
    op, children = ast
    if op == T.TOK["MAP_COLOR"]:
        src = _tok_to_num(children[0])
        dst = _tok_to_num(children[1])
        return OP.map_color(grid, src, dst)
    if op == T.TOK["FILTER_EQ"]:
        col = _tok_to_num(children[0])
        return OP.filter_eq(grid, col)
    if op == T.TOK["ROTATE"]:
        k = _tok_to_num(children[0])
        return OP.rotate(grid, k)
    if op == T.TOK["REFLECT"]:
        t = _tok_to_num(children[0])
        return OP.reflect(grid, t)
    if op == T.TOK["CROP"]:
        top = _tok_to_num(children[0]); left = _tok_to_num(children[1])
        h = _tok_to_num(children[2]); w = _tok_to_num(children[3])
        return OP.crop(grid, top, left, h, w)
    if op == T.TOK["PAD"]:
        top = _tok_to_num(children[0]); bottom = _tok_to_num(children[1])
        left = _tok_to_num(children[2]); right = _tok_to_num(children[3])
        c = _tok_to_num(children[4])
        return OP.pad(grid, top, bottom, left, right, c)
    if op == T.TOK["COMPOSE"]:
        # children: [progA, progB] as ASTs
        a = _exec_ast(children[0], grid)
        b = _exec_ast(children[1], a)
        return b
    # Extended ops
    if op == T.TOK["TRANSLATE"]:
        dy = _tok_to_num(children[0]); dx = _tok_to_num(children[1])
        return OP.translate(grid, dy, dx)
    if op == T.TOK["DRAW_LINE"]:
        y0 = _tok_to_num(children[0]); x0 = _tok_to_num(children[1])
        y1 = _tok_to_num(children[2]); x1 = _tok_to_num(children[3])
        col = _tok_to_num(children[4])
        return OP.draw_line(grid, y0, x0, y1, x1, col)
    if op == T.TOK["FILL_RECT"]:
        top = _tok_to_num(children[0]); left = _tok_to_num(children[1])
        h = _tok_to_num(children[2]); w = _tok_to_num(children[3])
        col = _tok_to_num(children[4])
        return OP.fill_rect(grid, top, left, h, w, col)
    if op == T.TOK["REPEAT_TILE"]:
        ny = _tok_to_num(children[0]); nx = _tok_to_num(children[1])
        return OP.repeat_tile(grid, ny, nx)
    if op == T.TOK["GET_BBOX"]:
        col = _tok_to_num(children[0])
        return OP.get_bbox(grid, col)
    if op == T.TOK["PAINT_OBJECT"]:
        src = _tok_to_num(children[0]); dst = _tok_to_num(children[1])
        return OP.paint_object(grid, src, dst)
    if op == T.TOK["FIND_COMPONENTS"]:
        off = _tok_to_num(children[0])
        return OP.find_components(grid, off)
    if op == T.TOK["COUNT_COLOR"]:
        col = _tok_to_num(children[0])
        return OP.count_color(grid, col)
    if op == T.TOK["MAJORITY_COLOR"]:
        tie = _tok_to_num(children[0])
        return OP.majority_color(grid, tie)
    if op == T.TOK["OVERLAY_UNION"]:
        a = _tok_to_num(children[0]); b = _tok_to_num(children[1]); oc = _tok_to_num(children[2])
        return OP.overlay_union(grid, a, b, oc)
    if op == T.TOK["OVERLAY_INTERSECT"]:
        a = _tok_to_num(children[0]); b = _tok_to_num(children[1]); oc = _tok_to_num(children[2])
        return OP.overlay_intersect(grid, a, b, oc)
    raise ValueError(f"Unknown op token: {op}")


def run_tokens(tokens: List[int], grid: np.ndarray) -> Optional[np.ndarray]:
    from app.dsl.program import tokens_to_ast

    ast = tokens_to_ast(tokens)
    if ast is None:
        return None
    out = _exec_ast(ast, grid)
    return out


def consistent_on_pairs(tokens: List[int], pairs: List[dict]) -> bool:
    for pair in pairs:
        out = run_tokens(tokens, pair["input"])
        if out is None or out.shape != pair["output"].shape:
            return False
        if not np.array_equal(out, pair["output"]):
            return False
    return True
