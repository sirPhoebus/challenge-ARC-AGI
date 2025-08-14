from __future__ import annotations
from typing import Callable, Dict, List, Tuple, Optional, Iterable
import numpy as np
from app import config
from app.dsl.interpreter import run_tokens
from app.dsl.program import valid_tokens

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


def beam_search(pairs: List[dict], enumerator: Optional[Iterable[List[int]]] = None,
                scorer: Optional[Callable[[List[int]], float]] = None,
                max_nodes: Optional[int] = None, beam_width: Optional[int] = None) -> Tuple[Optional[List[int]], float]:
    max_nodes = max_nodes or int(config.SEARCH_CFG["max_nodes"])
    beam_width = beam_width or int(config.SEARCH_CFG["beam_width"])

    if enumerator is None:
        from app.search.enumerator import enumerate_programs
        enumerator = enumerate_programs(config.DSL_MAX_DEPTH)

    beam: List[Tuple[List[int], float]] = []
    best: Tuple[Optional[List[int]], float] = (None, float(config.SEARCH_BAD_SCORE))
    cache_on = bool(config.SEARCH_CFG.get("cache_results", True))
    score_cache: Dict[Tuple[int, ...], float] = {} if cache_on else {}
    n = 0
    for tokens in enumerator:
        if not valid_tokens(tokens, config.DSL_MAX_DEPTH):
            continue
        tkey = tuple(tokens)
        if cache_on and tkey in score_cache:
            score = score_cache[tkey]
        else:
            score = scorer(tokens) if scorer is not None else _heuristic_score(tokens, pairs)
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
