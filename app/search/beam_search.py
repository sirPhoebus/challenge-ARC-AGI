from __future__ import annotations
from typing import Callable, Dict, List, Tuple, Optional, Iterable, Union
import numpy as np
import torch
import torch.nn.functional as F
from app import config
from app.dsl.interpreter import run_tokens
from app.dsl.program import valid_tokens
from app.utils import tokens as T
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


# ----------------------------
# Typed, execution-guided incremental prefix search
# ----------------------------

def _arg_choices_for(op_tok: int) -> List[List[int]]:
    """Return allowed token choices for each arg position of op_tok.
    Mirrors logic in app/search/enumerator.py but local to avoid private access.
    """
    NUMS = list(config.DSL_ENUM_NUMS)
    COLORS = list(range(config.DSL_NUM_COLORS))
    ROTS = list(config.DSL_ROTATIONS)
    REFS = list(config.DSL_REFLECTIONS)
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
    # Extended ops
    if op_tok == T.TOK["FIND_COMPONENTS"]:
        nums = [T.NUM_TOKENS[n] for n in NUMS]
        return [nums]
    if op_tok == T.TOK["GET_BBOX"]:
        cols = [T.COLOR_TOKENS[c] for c in COLORS]
        return [cols]
    if op_tok == T.TOK["PAINT_OBJECT"]:
        cols = [T.COLOR_TOKENS[c] for c in COLORS]
        return [cols, cols]
    if op_tok == T.TOK["COUNT_COLOR"]:
        cols = [T.COLOR_TOKENS[c] for c in COLORS]
        return [cols]
    if op_tok == T.TOK["MAJORITY_COLOR"]:
        cols = [T.COLOR_TOKENS[c] for c in COLORS]
        return [cols]
    if op_tok == T.TOK["TRANSLATE"]:
        nums = [T.NUM_TOKENS[n] for n in NUMS]
        return [nums, nums]
    if op_tok == T.TOK["DRAW_LINE"]:
        nums = [T.NUM_TOKENS[n] for n in NUMS]
        cols = [T.COLOR_TOKENS[c] for c in COLORS]
        return [nums, nums, nums, nums, cols]
    if op_tok == T.TOK["FILL_RECT"]:
        nums = [T.NUM_TOKENS[n] for n in NUMS]
        cols = [T.COLOR_TOKENS[c] for c in COLORS]
        return [nums, nums, nums, nums, cols]
    if op_tok == T.TOK["REPEAT_TILE"]:
        nums = [T.NUM_TOKENS[n] for n in NUMS]
        return [nums, nums]
    if op_tok == T.TOK["OVERLAY_UNION"]:
        cols = [T.COLOR_TOKENS[c] for c in COLORS]
        return [cols, cols, cols]
    if op_tok == T.TOK["OVERLAY_INTERSECT"]:
        cols = [T.COLOR_TOKENS[c] for c in COLORS]
        return [cols, cols, cols]
    return []


class _Expect:
    # Expectation for parser stack
    def __init__(self, kind: str, parent_op: Optional[int] = None, is_left: Optional[bool] = None,
                 start_idx: Optional[int] = None, arg_choices: Optional[List[int]] = None):
        self.kind = kind  # 'program' or 'arg'
        self.parent_op = parent_op
        self.is_left = is_left
        self.start_idx = start_idx  # where this program starts in token stream
        self.arg_choices = arg_choices or []


def _typed_next(prefix: List[int]) -> Tuple[bool, Union[str, None], List[int], List[Tuple[int, int, bool]]]:
    """Parse prefix and compute next expectation and allowed tokens.
    Returns:
      ok: whether prefix is syntactically valid so far
      need: 'op', 'arg', or None if complete
      allowed: list of allowed token ids for next position (empty if None)
      completed: list of completed subprogram spans [(start, end, is_left_child)]
    """
    stack: List[_Expect] = [_Expect('program', parent_op=None, is_left=None, start_idx=0)]
    i = 0
    completed: List[Tuple[int, int, bool]] = []
    while i < len(prefix):
        if not stack:
            return False, None, [], completed
        top = stack.pop()
        if top.kind == 'program':
            tok = prefix[i]
            i += 1
            if tok not in T.OP_ARITY:
                return False, None, [], completed
            k = int(T.OP_ARITY[tok])
            if tok == T.TOK["COMPOSE"]:
                # push right then left (so left is processed next)
                stack.append(_Expect('program', parent_op=tok, is_left=False, start_idx=i))
                stack.append(_Expect('program', parent_op=tok, is_left=True, start_idx=i))
            else:
                # push args (rightmost first)
                arg_lists = _arg_choices_for(tok)
                for al in reversed(arg_lists):
                    stack.append(_Expect('arg', parent_op=tok, is_left=None, start_idx=None, arg_choices=al))
        else:
            # expecting argument
            tok = prefix[i]
            i += 1
            if tok not in top.arg_choices:
                return False, None, [], completed
        # Detect completed subprograms: whenever next on stack is not an 'arg' for this current op
        # we mark completion when we just finished satisfying a 'program' frame.
        # We approximate by checking if next expected is a 'program' whose start_idx is set and
        # i equals its start (we just finished the previous one). To be robust, track when we pop
        # into a state where top of stack is a pending 'program' and the previous consumed was a program end.
        # Here, we detect a completion when the next expectation is 'program' and the token we just consumed
        # was not an op for its args.
        # For simplicity, re-derive completion spans by a secondary pass: attempt to parse from each
        # position where a 'program' starts to see if it ends exactly at i. We'll handle only left-child
        # completions (is_left==True).
        # Build a quick check: if stack and stack[-1].kind == 'program' and stack[-1].is_left is False,
        # then the left sibling just completed covering a span ending at i.
        if stack and stack[-1].kind == 'program' and stack[-1].is_left is False:
            # left child span is from the previous 'program' start to i
            # Find the last left child start
            # We search backward for the most recent _Expect with is_left True having start_idx set
            for j in range(len(stack) - 1, -1, -1):
                ex = stack[j]
                if ex.kind == 'program' and ex.is_left is True and ex.start_idx is not None and ex.start_idx < i:
                    completed.append((ex.start_idx, i, True))
                    break
    # After consuming prefix
    if not stack:
        return True, None, [], completed
    top = stack[-1]
    if top.kind == 'program':
        # expect an op next
        allowed = [tok for tok in T.ENABLED_OP_TOKENS]
        return True, 'op', allowed, completed
    else:
        # expect an argument from provided choices
        return True, 'arg', list(top.arg_choices), completed


def _exec_heuristic(tokens: List[int], pairs: List[dict]) -> Optional[float]:
    # Returns negative mean Hamming distance; None if invalid execution
    dists = []
    for pr in pairs:
        out = run_tokens(tokens, pr["input"])
        if out is None:
            return None
        tgt = pr["output"]
        if getattr(out, 'shape', None) != getattr(tgt, 'shape', None):
            return float(config.SEARCH_BAD_SCORE)
        if out.size == 0:
            return float(config.SEARCH_BAD_SCORE)
        dists.append(float(np.mean(out != tgt)))
    if not dists:
        return 0.0
    return -float(np.mean(dists))


def incremental_beam_search(
    pairs: List[dict],
    policy,  # ProgramPolicy instance
    beam_width: Optional[int] = None,
    max_nodes: Optional[int] = None,
    time_budget_s: Optional[float] = None,
) -> Tuple[Optional[List[int]], float]:
    """Typed incremental beam search guided by policy logits and execution.
    Score = w_logprob * sum_logprob + w_heur * heur_exec.
    """
    s_cfg = config.SEARCH_CFG
    bw = int(beam_width or s_cfg.get("policy_beam_width", s_cfg.get("beam_width", 16)))
    maxn = int(max_nodes or s_cfg.get("policy_max_nodes", s_cfg.get("max_nodes", 1000)))
    w_lp = float(s_cfg.get("typed_w_logprob", 1.0))
    w_h = float(s_cfg.get("typed_w_heur", 1.0))
    per_node = int(s_cfg.get("max_expansions_per_node", 32))
    cache_on = bool(s_cfg.get("cache_results", True))
    exec_cache: Dict[Tuple[int, ...], float] = {} if cache_on else {}
    start_t = time.perf_counter()
    use_subset = bool(s_cfg.get("pair_subset_enable", False))
    hpairs = _select_pairs(pairs) if use_subset else pairs

    class State:
        __slots__ = ("tokens", "sum_lp", "score", "last_exec_tokens")

        def __init__(self, tokens=None, sum_lp=0.0, score=float(config.SEARCH_BAD_SCORE), last_exec_tokens=None):
            self.tokens: List[int] = tokens or []
            self.sum_lp: float = float(sum_lp)
            self.score: float = float(score)
            self.last_exec_tokens: Optional[List[int]] = last_exec_tokens  # last executed prefix tokens

    beam: List[State] = [State(tokens=[], sum_lp=0.0, score=0.0, last_exec_tokens=None)]
    best: Tuple[Optional[List[int]], float] = (None, float(config.SEARCH_BAD_SCORE))
    expanded = 0

    while beam and expanded < maxn:
        if time_budget_s is not None and time_budget_s > float(config.INFER_CFG.get("time_budget_disable_threshold", 0.0)):
            if (time.perf_counter() - start_t) >= float(time_budget_s):
                break
        new_beam: List[State] = []
        # Expand each state
        for st in beam:
            ok, need, allowed, completed = _typed_next(st.tokens)
            if not ok:
                continue
            if need is None:
                # Complete program: score exactly and consider best
                heur = _exec_heuristic(st.tokens, hpairs)
                if heur is None:
                    continue
                total = w_lp * st.sum_lp + w_h * heur
                if total > best[1]:
                    best = (st.tokens, total)
                # Do not expand further
                continue
            # Query policy for next-token logits
            with torch.no_grad():
                tok_tensor = torch.tensor(st.tokens, dtype=torch.long)
                policy_pairs = [(pr["input"], pr["output"]) for pr in hpairs]
                logits, _ = policy(policy_pairs, tok_tensor)
                logp = F.log_softmax(logits, dim=-1).cpu().numpy()
            # Mask to allowed set
            if not allowed:
                continue
            mask = np.full_like(logp, -np.inf)
            mask[np.array(allowed, dtype=np.int64)] = 0.0
            scores = logp + mask
            # Top-k per node
            idxs = np.argsort(scores)[::-1][:per_node]
            for idx in idxs:
                if not np.isfinite(scores[idx]):
                    continue
                new_tokens = st.tokens + [int(idx)]
                new_sum_lp = st.sum_lp + float(scores[idx])
                nst = State(tokens=new_tokens, sum_lp=new_sum_lp, score=st.score, last_exec_tokens=st.last_exec_tokens)
                # If we just completed a left child of COMPOSE, run execution for pruning and scoring
                ok2, _, _, completed2 = _typed_next(new_tokens)
                if not ok2:
                    continue
                updated = False
                if completed2:
                    # pick the most recent completed left child span
                    start_idx, end_idx, is_left = completed2[-1]
                    if is_left and 0 <= start_idx < end_idx <= len(new_tokens):
                        left_span = new_tokens[start_idx:end_idx]
                        # Compose previous executed prefix (if any) with this left span
                        exec_toks = left_span if not nst.last_exec_tokens else [T.TOK["COMPOSE"]] + nst.last_exec_tokens + left_span
                        key = tuple(exec_toks)
                        if cache_on and key in exec_cache:
                            heur = exec_cache[key]
                        else:
                            heur = _exec_heuristic(exec_toks, hpairs)
                            if heur is None:
                                # prune invalid
                                continue
                            if cache_on:
                                exec_cache[key] = heur
                        nst.last_exec_tokens = exec_toks
                        # Update running score proxy to help sort beam
                        nst.score = w_lp * nst.sum_lp + w_h * heur
                        updated = True
                if not updated:
                    # keep previous score
                    nst.score = w_lp * nst.sum_lp + w_h * (nst.score if np.isfinite(nst.score) else 0.0)
                new_beam.append(nst)
                expanded += 1
                if expanded >= maxn:
                    break
            if expanded >= maxn:
                break
        # Prune beam
        new_beam.sort(key=lambda s: s.score, reverse=True)
        beam = new_beam[:bw]
        if not beam:
            break
    return best
