from __future__ import annotations
from typing import Iterable, List, Optional
from app import config
from app.utils import tokens as T

# Size-by-size enumerator with type-safe expansion and pruning.

NUMS = list(config.DSL_ENUM_NUMS)
COLORS = list(range(config.DSL_NUM_COLORS))
ROTS = list(config.DSL_ROTATIONS)
REFS = list(config.DSL_REFLECTIONS)


def _args_for(op_tok: int) -> List[List[int]]:
    # Return list of lists of valid argument token choices for an op (non-compose ops).
    if op_tok == T.TOK["MAP_COLOR"]:
        return [[T.COLOR_TOKENS[c] for c in COLORS], [T.COLOR_TOKENS[c] for c in COLORS]]
    if op_tok == T.TOK["FILTER_EQ"]:
        return [[T.COLOR_TOKENS[c] for c in COLORS]]
    if op_tok == T.TOK["ROTATE"]:
        return [[T.ROT_TOKENS[k] for k in ROTS]]
    if op_tok == T.TOK["REFLECT"]:
        return [[T.REFLECT_TOKENS[t] for t in REFS]]
    if op_tok == T.TOK["CROP"]:
        nums = [T.NUM_TOKENS[n] for n in NUMS]
        return [nums, nums, nums, nums]
    if op_tok == T.TOK["PAD"]:
        nums = [T.NUM_TOKENS[n] for n in NUMS]
        cols = [T.COLOR_TOKENS[c] for c in COLORS]
        return [nums, nums, nums, nums, cols]
    return []


def enumerate_programs(max_depth: Optional[int] = None) -> Iterable[List[int]]:
    max_depth = max_depth or int(config.DSL_MAX_DEPTH)
    base_ops = [tok for tok in T.ENABLED_OP_TOKENS if tok != T.TOK["COMPOSE"]]

    # Depth-1: single primitive with args
    for op in base_ops:
        arg_choices = _args_for(op)
        if not arg_choices:
            continue
        # Cartesian product over args, but limit per-node expansions
        idx_limits = [len(xs) for xs in arg_choices]
        total = 1
        for L in idx_limits:
            total *= L
        limit = int(config.SEARCH_CFG["max_expansions_per_node"]) or total
        count = 0
        def rec(i: int, acc: List[int]):
            nonlocal count
            if i == len(arg_choices):
                count += 1
                if count <= limit:
                    yield [op] + acc
                return
            for tok in arg_choices[i]:
                if count >= limit:
                    break
                for y in rec(i + 1, acc + [tok]):
                    yield y
        for seq in rec(0, []):
            yield seq

    # Depth>=2: compose smaller programs
    if max_depth >= 2:
        # Cache programs by depth
        by_depth: List[List[List[int]]] = [[] for _ in range(max_depth + 1)]
        # fill depth 1
        for op in base_ops:
            arg_choices = _args_for(op)
            if not arg_choices:
                continue
            idx_limits = [len(xs) for xs in arg_choices]
            total = 1
            for L in idx_limits:
                total *= L
            limit = int(config.SEARCH_CFG["max_expansions_per_node"]) or total
            count = 0
            def rec2(i: int, acc: List[int]):
                nonlocal count
                if i == len(arg_choices):
                    count += 1
                    if count <= limit:
                        by_depth[1].append([op] + acc)
                    return
                for tok in arg_choices[i]:
                    if count >= limit:
                        break
                    rec2(i + 1, acc + [tok])
            rec2(0, [])

        # build higher depths by composing any two <= d-1 that sum to d
        for d in range(2, max_depth + 1):
            for d1 in range(1, d):
                for a in by_depth[d1]:
                    for b in by_depth[d - d1]:
                        seq = [T.TOK["COMPOSE"]] + a + b
                        yield seq
                        by_depth[d].append(seq)
