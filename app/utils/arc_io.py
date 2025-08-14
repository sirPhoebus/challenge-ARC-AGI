from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
from app import config

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_tasks(split: str = "training", limit: Optional[int] = None) -> List[Dict[str, Any]]:
    folder = os.path.join(DATA_DIR, split)
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".json")]
    files.sort()
    if limit is not None:
        files = files[:limit]
    tasks: List[Dict[str, Any]] = []
    for p in files:
        task = _read_json(p)
        # Normalize to numpy int32 grids
        def to_np(g: List[List[int]]) -> np.ndarray:
            a = np.array(g, dtype=np.int32)
            return a
        for k in ["train", "test"]:
            if k in task:
                task[k] = [
                    {
                        "input": to_np(pair["input"] if isinstance(pair, dict) else pair[0]),
                        "output": to_np(pair["output"] if isinstance(pair, dict) else pair[1]),
                    }
                    for pair in task[k]
                ]
        tasks.append(task)
    return tasks


def clamp_color_grid(a: np.ndarray) -> np.ndarray:
    # Ensure values within 0..num_colors-1
    return np.clip(a, 0, int(config.DSL_NUM_COLORS) - 1).astype(np.int32)
