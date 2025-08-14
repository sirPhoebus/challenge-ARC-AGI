from __future__ import annotations
from typing import Callable, Dict, List, Tuple, Optional, Iterable
import numpy as np
from app import config
from app.dsl.interpreter import run_tokens
from app.dsl.program import valid_tokens
import time

# Simple beam search over token-seq programs. Optionally guided by a policy scorer.
# scorer(partial_tokens) -> float (higher is better). If None, use heuristic.


def _heuristic_score(tokens: List[int], pairs: List[dict]) -> float:
    # Score = negative average Hamming distance (after resizing mismatch -> -inf)
    dists = []
    for pr in pairs:
        out = run_tokens(tokens, pr["input"])
        if out is None or out.shape != pr["output"].shape:
            return float(config.SEARCH_BAD_SCORE)
        dists.append(float(np.mean(out != pr["output"])) if out.size > 0 else 1.0)
    return -float(np.mean(dists))


def _select_pairs(pairs: List[dict]) -> List[dict]:
    """Select a subset of I/O pairs for heuristic scoring based on config.
    Strategy "entropy_size": prefer higher color entropy then larger grids.
    """
    if not pairs:
        return pairs
    try:
        enable = bool(config.SEARCH_CFG["pair_subset_enable"])
        k = int(config.SEARCH_CFG["pair_subset_k"])
        strat = str(config.SEARCH_CFG["pair_subset_strategy"])
    except Exception:
        return pairs
    if (not enable) or k >= len(pairs):
        return pairs

    def entropy(a: np.ndarray) -> float:
        vals, counts = np.unique(a, return_counts=True)
        p = counts.astype(np.float64)
        p = p / max(1, p.sum())
        # Use natural log; relative ordering unaffected by log base
        eps = float(config.SEARCH_CFG["entropy_eps"])
        cmax = float(config.SEARCH_CFG["entropy_clip_max"])
        return float(-np.sum(p * np.log(np.clip(p, eps, cmax))))

    scored: List[Tuple[float, float, int]] = []
    for i, pr in enumerate(pairs):
        grid = pr.get("input")
        h, w = (getattr(grid, "shape", (0, 0)) or (0, 0))
        size = float(h) * float(w)
        ent = entropy(grid)
        # sort key: primary entropy desc, then size desc
        scored.append((ent, size, i))

    if strat == "entropy_size":
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    else:
        # default fallback: size only
        scored.sort(key=lambda t: (t[1], t[0]), reverse=True)
    idxs = [i for _, _, i in scored[:k]]
    return [pairs[i] for i in idxs]


def beam_search(pairs: List[dict], enumerator: Optional[Iterable[List[int]]] = None,
                scorer: Optional[Callable[[List[int]], float]] = None,
                max_nodes: Optional[int] = None, beam_width: Optional[int] = None,
                time_budget_s: Optional[float] = None) -> Tuple[Optional[List[int]], float]:
    max_nodes = max_nodes or int(config.SEARCH_CFG["max_nodes"])
    beam_width = beam_width or int(config.SEARCH_CFG["beam_width"])

    if enumerator is None:
        from app.search.enumerator import enumerate_programs
        enumerator = enumerate_programs(config.DSL_MAX_DEPTH)

    # Apply pair subset routing for heuristic-only mode
    hpairs = pairs
    if scorer is None and bool(config.SEARCH_CFG["pair_subset_enable"]):
        hpairs = _select_pairs(pairs)

    beam: List[Tuple[List[int], float]] = []
    best: Tuple[Optional[List[int]], float] = (None, float(config.SEARCH_BAD_SCORE))
    cache_on = bool(config.SEARCH_CFG.get("cache_results", True))
    score_cache: Dict[Tuple[int, ...], float] = {} if cache_on else {}
    n = 0
    start_t = time.perf_counter()
    for tokens in enumerator:
        if not valid_tokens(tokens, config.DSL_MAX_DEPTH):
            continue
        if time_budget_s is not None and time_budget_s > float(config.INFER_CFG["time_budget_disable_threshold"]):
            if (time.perf_counter() - start_t) >= float(time_budget_s):
                break
        tkey = tuple(tokens)
        if cache_on and tkey in score_cache:
            score = score_cache[tkey]
        else:
            score = scorer(tokens) if scorer is not None else _heuristic_score(tokens, hpairs)
            if cache_on:
                score_cache[tkey] = score
        n += 1
        if score > best[1]:
            best = (tokens, score)
        # maintain beam
        beam.append((tokens, score))
        beam.sort(key=lambda x: x[1], reverse=True)
        if len(beam) > beam_width:
            beam = beam[:beam_width]
        if n >= max_nodes:
            break
    return best
