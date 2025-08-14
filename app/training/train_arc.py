from __future__ import annotations
import json
import os
import random
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from app import config
from app.utils import arc_io
from app.utils import tokens as T
from app.dsl.interpreter import run_tokens
from app.search.enumerator import enumerate_programs
from app.model.program_policy import ProgramPolicy

STATE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "state")
PSEUDO_PATH = os.path.join(STATE_DIR, "pseudo_labels.jsonl")
CKPT_PATH = os.path.join(STATE_DIR, "policy.pt")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collect_pseudo_labels() -> List[Dict]:
    tasks = arc_io.load_tasks("training", limit=int(config.TRAIN_CFG["task_limit"]))
    max_depth = int(config.TRAIN_CFG["pseudo_max_depth"]) or int(config.DSL_MAX_DEPTH)
    samples: List[Dict] = []
    for task in tasks:
        pairs = task["train"]
        for tokens in enumerate_programs(max_depth):
            ok = True
            for pr in pairs:
                out = run_tokens(tokens, pr["input"])
                if out is None or out.shape != pr["output"].shape or not np.array_equal(out, pr["output"]):
                    ok = False
                    break
            if ok:
                # store all prefixes as teacher forcing examples
                for i in range(1, len(tokens)):
                    samples.append({
                        "task_id": task.get("id", "unknown"),
                        "pairs": [
                            {"input": pr["input"].tolist(), "output": pr["output"].tolist()} for pr in pairs
                        ],
                        "prefix": tokens[:i],
                        "next": tokens[i],
                        "final": 1 if i == len(tokens) - 1 else 0,
                    })
                # also store an END token target for full program (optional)
                samples.append({
                    "task_id": task.get("id", "unknown"),
                    "pairs": [
                        {"input": pr["input"].tolist(), "output": pr["output"].tolist()} for pr in pairs
                    ],
                    "prefix": tokens,
                    "next": T.TOK["END"],
                    "final": 1,
                })
                break  # one solution per task to keep it small
    return samples


def save_jsonl(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def train():
    seed_all(int(config.TRAIN_CFG["seed"]))
    if not os.path.exists(PSEUDO_PATH):
        samples = collect_pseudo_labels()
        save_jsonl(PSEUDO_PATH, samples)
    else:
        samples = load_jsonl(PSEUDO_PATH)

    model = ProgramPolicy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=float(config.TRAIN_CFG["lr"]), weight_decay=float(config.TRAIN_CFG["weight_decay"]))
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    def to_tensor_grid(g):
        return torch.tensor(np.array(g, dtype=np.int64), dtype=torch.long, device=device)

    B = int(config.TRAIN_CFG["batch_size"]) or 32
    epochs = int(config.TRAIN_CFG["epochs"]) or 1

    for ep in range(epochs):
        random.shuffle(samples)
        total_loss = 0.0
        n = 0
        for i in range(0, len(samples), B):
            batch = samples[i : i + B]
            opt.zero_grad()
            loss = 0.0
            for row in batch:
                pairs = [(to_tensor_grid(p["input"]), to_tensor_grid(p["output"])) for p in row["pairs"]]
                prefix = torch.tensor(row["prefix"], dtype=torch.long, device=device)
                target = torch.tensor(row["next"], dtype=torch.long, device=device)
                logits, val = model(pairs, prefix)
                loss = loss + ce(logits.unsqueeze(0), target.unsqueeze(0)) + mse(val, torch.tensor(float(row["final"]), device=device))
            loss = loss / max(1, len(batch))
            loss.backward()
            if float(config.TRAIN_CFG["grad_clip"]) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), float(config.TRAIN_CFG["grad_clip"]))
            opt.step()
            total_loss += float(loss.item())
            n += 1
        print(f"epoch {ep+1}/{epochs} | batches={n} | loss={total_loss/max(1,n):.4f}")

    torch.save(model.state_dict(), CKPT_PATH)
    print(f"Saved policy to {CKPT_PATH}")


if __name__ == "__main__":
    train()
