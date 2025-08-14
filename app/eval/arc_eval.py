from __future__ import annotations
import os
from typing import Callable, List, Optional, Tuple
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed

from app import config
from app.utils import arc_io
from app.dsl.interpreter import consistent_on_pairs
from app.search.beam_search import beam_search, incremental_beam_search
from app.search.enumerator import enumerate_programs

STATE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "state")
CKPT_PATH = os.path.join(STATE_DIR, "policy.pt")


def _load_policy_model() -> Optional["ProgramPolicy"]:
    # Load ProgramPolicy if checkpoint exists; return eval-mode model or None.
    if not os.path.exists(CKPT_PATH):
        return None
    try:
        from app.model.program_policy import ProgramPolicy
    except Exception:
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProgramPolicy().to(device)
    try:
        sd = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(sd)
        model.eval()
        return model
    except Exception:
        return None


def _solve_task_nopolicy(task: dict, max_depth: int, beam_width: int, max_nodes: int) -> int:
    """Solve a single task without policy guidance. Returns 1 if solved else 0.
    Top-level function for multiprocessing.
    """
    from app.search.enumerator import enumerate_programs  # local import for spawn safety
    from app.search.beam_search import beam_search
    from app.dsl.interpreter import consistent_on_pairs

    pairs = task.get("train", [])
    if not pairs:
        pairs = task.get("test", [])
    enum = enumerate_programs(int(max_depth))
    tb = float(config.INFER_CFG["time_budget_s"])
    best_tokens, best_score = beam_search(
        pairs,
        enumerator=enum,
        scorer=None,
        max_nodes=max_nodes,
        beam_width=beam_width,
        time_budget_s=tb,
    )
    return 1 if (best_tokens is not None and consistent_on_pairs(best_tokens, pairs)) else 0


def evaluate(limit_tasks: Optional[int] = None) -> Tuple[int, int]:
    tasks = arc_io.load_tasks("evaluation", limit=limit_tasks)
    solved = 0
    total = len(tasks)

    policy_model = _load_policy_model()

    bw = int(config.INFER_CFG.get("beam_width", config.SEARCH_CFG["beam_width"]))
    mn = int(config.INFER_CFG.get("max_nodes", config.SEARCH_CFG["max_nodes"]))
    nw = int(config.INFER_CFG["num_workers"])
    use_policy = bool(config.INFER_CFG["use_policy"]) and (policy_model is not None)
    time_budget = float(config.INFER_CFG["time_budget_s"])

    # If using policy, optionally tighten/modify beam settings from SEARCH config
    eff_bw = int(config.SEARCH_CFG.get("policy_beam_width", bw)) if use_policy else bw
    eff_mn = int(config.SEARCH_CFG.get("policy_max_nodes", mn)) if use_policy else mn

    # Parallel path only when NOT using policy (to avoid GPU contention)
    if (not use_policy) and nw > 1 and total > 1:
        print(f"[eval] parallel mode: workers={nw} | beam={bw} | max_nodes={mn} | time_budget_s={time_budget}", flush=True)
        with ProcessPoolExecutor(max_workers=nw) as ex:
            futures = [ex.submit(_solve_task_nopolicy, task, int(config.DSL_MAX_DEPTH), bw, mn) for task in tasks]
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    solved += int(fut.result())
                except Exception:
                    pass
                if i % 1 == 0:
                    print(f"[eval] completed {i}/{total} | solved={solved}", flush=True)
    else:
        if use_policy and nw > 1:
            print(f"[eval] policy enabled -> running single-process to use GPU efficiently (requested workers={nw})", flush=True)
        for idx, task in enumerate(tasks):
            print(f"[eval] task {idx+1}/{total} ...", flush=True)
            pairs = task.get("train", [])
            if not pairs:
                pairs = task.get("test", [])
            if use_policy and policy_model is not None:
                best_tokens, best_score = incremental_beam_search(
                    pairs,
                    policy_model,
                    beam_width=eff_bw,
                    max_nodes=eff_mn,
                    time_budget_s=time_budget,
                )
            else:
                enum = enumerate_programs(int(config.DSL_MAX_DEPTH))
                best_tokens, best_score = beam_search(
                    pairs,
                    enumerator=enum,
                    scorer=None,
                    max_nodes=eff_mn,
                    beam_width=eff_bw,
                    time_budget_s=time_budget,
                )
            if best_tokens is not None and consistent_on_pairs(best_tokens, pairs):
                solved += 1
    print(f"Solved {solved}/{total} evaluation tasks")
    return solved, total


if __name__ == "__main__":
    evaluate()
