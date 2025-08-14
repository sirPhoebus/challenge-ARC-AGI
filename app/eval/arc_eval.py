from __future__ import annotations
import os
from typing import Callable, List, Optional, Tuple
import numpy as np
import torch

from app import config
from app.utils import arc_io
from app.dsl.interpreter import consistent_on_pairs
from app.search.beam_search import beam_search
from app.search.enumerator import enumerate_programs

STATE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "state")
CKPT_PATH = os.path.join(STATE_DIR, "policy.pt")


def _make_policy_scorer() -> Optional[Callable[[List[int], List[dict]], float]]:
    # Returns a scorer(tokens, pairs) -> float using the ProgramPolicy value head, if checkpoint exists.
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
    except Exception:
        return None

    def to_tensor_grid(g: np.ndarray) -> torch.Tensor:
        return torch.tensor(np.array(g, dtype=np.int64), dtype=torch.long, device=device)

    @torch.no_grad()
    def scorer(tokens: List[int], pairs: List[dict]) -> float:
        # Use value head as score estimate for the current prefix.
        ppairs = [(to_tensor_grid(p["input"]), to_tensor_grid(p["output"])) for p in pairs]
        prefix = torch.tensor(tokens, dtype=torch.long, device=device)
        logits, val = model(ppairs, prefix)
        return float(val.item())

    # Wrap to match beam_search scorer signature: scorer(tokens) -> float, capture pairs later.
    def wrap_factory(pairs: List[dict]) -> Callable[[List[int]], float]:
        return lambda toks: scorer(toks, pairs)

    # We return a factory; caller binds pairs per task.
    return wrap_factory  # type: ignore[return-value]


def evaluate(limit_tasks: Optional[int] = None) -> Tuple[int, int]:
    tasks = arc_io.load_tasks("evaluation", limit=limit_tasks)
    solved = 0
    total = len(tasks)

    factory = _make_policy_scorer()

    for task in tasks:
        pairs = task.get("train", [])
        if not pairs:
            pairs = task.get("test", [])
        if factory is not None:
            scorer = factory(pairs)
        else:
            scorer = None  # type: ignore[assignment]

        enum = enumerate_programs(int(config.DSL_MAX_DEPTH))
        best_tokens, best_score = beam_search(pairs, enumerator=enum, scorer=scorer)
        if best_tokens is not None and consistent_on_pairs(best_tokens, pairs):
            solved += 1
    print(f"Solved {solved}/{total} evaluation tasks")
    return solved, total


if __name__ == "__main__":
    evaluate()
