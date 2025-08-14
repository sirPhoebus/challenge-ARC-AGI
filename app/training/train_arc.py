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

try:
    from tqdm import tqdm as _tqdm  # optional dependency, controlled via config
except Exception:
    _tqdm = None

STATE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "state")
PSEUDO_PATH = os.path.join(STATE_DIR, "pseudo_labels.jsonl")
CKPT_PATH = os.path.join(STATE_DIR, "policy.pt")
FULL_CKPT_PATH = os.path.join(STATE_DIR, "policy_full.pt")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collect_pseudo_labels() -> List[Dict]:
    tasks = arc_io.load_tasks("training", limit=int(config.TRAIN_CFG["task_limit"]))
    max_depth = int(config.TRAIN_CFG["pseudo_max_depth"]) or int(config.DSL_MAX_DEPTH)
    log_every = int(config.TRAIN_CFG.get("log_interval", 10))
    verbose = bool(config.TRAIN_CFG.get("verbose", True))
    use_tqdm = bool(config.TRAIN_CFG.get("use_tqdm", True)) and (_tqdm is not None)
    tqdm_mininterval = float(config.TRAIN_CFG.get("tqdm_mininterval", 0.1))
    tqdm_leave = bool(config.TRAIN_CFG.get("tqdm_leave", False))
    if verbose:
        print(f"[collect] tasks={len(tasks)} max_depth={max_depth}", flush=True)
    samples: List[Dict] = []
    solved_tasks = 0
    pbar = None
    if use_tqdm:
        pbar = _tqdm(total=len(tasks), desc="collect", mininterval=tqdm_mininterval, leave=tqdm_leave)
    for ti, task in enumerate(tasks):
        pairs = task["train"]
        found = False
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
                found = True
                break  # one solution per task to keep it small
        if found:
            solved_tasks += 1
        if use_tqdm and pbar is not None:
            pbar.update(1)
            pbar.set_postfix_str(f"solved={solved_tasks} samples={len(samples)}")
        elif verbose and ((ti + 1) % max(1, log_every) == 0 or (ti + 1) == len(tasks)):
            print(
                f"[collect] processed={ti+1}/{len(tasks)} solved={solved_tasks} samples={len(samples)}",
                flush=True,
            )
    if pbar is not None:
        pbar.close()
    if verbose:
        print(f"[collect] done: tasks={len(tasks)} solved={solved_tasks} samples={len(samples)}", flush=True)
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
    log_every = int(config.TRAIN_CFG.get("log_interval", 10))
    verbose = bool(config.TRAIN_CFG.get("verbose", True))
    use_tqdm = bool(config.TRAIN_CFG.get("use_tqdm", True)) and (_tqdm is not None)
    tqdm_mininterval = float(config.TRAIN_CFG.get("tqdm_mininterval", 0.1))
    tqdm_leave = bool(config.TRAIN_CFG.get("tqdm_leave", False))

    if verbose:
        print(
            f"[train] starting | seed={int(config.TRAIN_CFG['seed'])} | task_limit={int(config.TRAIN_CFG['task_limit'])} | pseudo_max_depth={int(config.TRAIN_CFG['pseudo_max_depth'])}",
            flush=True,
        )

    if not os.path.exists(PSEUDO_PATH):
        if verbose:
            print(f"[train] generating pseudo labels -> {PSEUDO_PATH}", flush=True)
        samples = collect_pseudo_labels()
        save_jsonl(PSEUDO_PATH, samples)
        if verbose:
            print(f"[train] generated samples={len(samples)}", flush=True)
    else:
        if verbose:
            print(f"[train] loading pseudo labels from {PSEUDO_PATH}", flush=True)
        samples = load_jsonl(PSEUDO_PATH)
        if verbose:
            print(f"[train] loaded samples={len(samples)}", flush=True)

    model = ProgramPolicy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=float(config.TRAIN_CFG["lr"]), weight_decay=float(config.TRAIN_CFG["weight_decay"]))
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    # Resume support
    start_epoch = 0
    if bool(config.TRAIN_CFG.get("resume", True)):
        loaded = False
        if os.path.exists(FULL_CKPT_PATH):
            try:
                ckpt = torch.load(FULL_CKPT_PATH, map_location=device)
                model.load_state_dict(ckpt.get("model_state", {}))
                if "optimizer_state" in ckpt:
                    opt.load_state_dict(ckpt["optimizer_state"])
                start_epoch = int(ckpt.get("epoch", 0))
                loaded = True
                if verbose:
                    print(f"[train] resumed full checkpoint from {FULL_CKPT_PATH} at epoch={start_epoch}", flush=True)
            except Exception as e:
                if verbose:
                    print(f"[train] failed to load full checkpoint: {e}", flush=True)
        if not loaded and os.path.exists(CKPT_PATH):
            try:
                sd = torch.load(CKPT_PATH, map_location=device)
                model.load_state_dict(sd)
                loaded = True
                if verbose:
                    print(f"[train] resumed weights from {CKPT_PATH}", flush=True)
            except Exception as e:
                if verbose:
                    print(f"[train] failed to load weights: {e}", flush=True)

    def to_tensor_grid(g):
        return torch.tensor(np.array(g, dtype=np.int64), dtype=torch.long, device=device)

    B = int(config.TRAIN_CFG["batch_size"]) or 32
    epochs = int(config.TRAIN_CFG["epochs"]) or 1
    if verbose:
        print(
            f"[train] device={device} cuda={torch.cuda.is_available()} | batch={B} | epochs={epochs}",
            flush=True,
        )

    save_every = int(config.TRAIN_CFG.get("save_every_epochs", 1))
    save_on_interrupt = bool(config.TRAIN_CFG.get("save_on_interrupt", True))

    try:
        for ep in range(start_epoch, epochs):
            random.shuffle(samples)
            total_loss = 0.0
            n = 0
            num_batches = (len(samples) + B - 1) // B
            batch_iter = range(0, len(samples), B)
            pbar = None
            if use_tqdm:
                pbar = _tqdm(total=num_batches, desc=f"train ep {ep+1}/{epochs}", mininterval=tqdm_mininterval, leave=tqdm_leave)
            for i in batch_iter:
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
                if use_tqdm and pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix_str(f"avg_loss={total_loss/max(1,n):.4f}")
                elif verbose and (n % max(1, log_every) == 0 or i + B >= len(samples)):
                    print(f"[train] epoch {ep+1}/{epochs} batch={n} avg_loss={total_loss/max(1,n):.4f}", flush=True)
            if pbar is not None:
                pbar.close()
            print(f"epoch {ep+1}/{epochs} | batches={n} | loss={total_loss/max(1,n):.4f}")

            # Periodic checkpointing
            if save_every > 0 and ((ep + 1) % save_every == 0):
                try:
                    torch.save(model.state_dict(), CKPT_PATH)
                    torch.save({
                        "model_state": model.state_dict(),
                        "optimizer_state": opt.state_dict(),
                        "epoch": ep + 1,
                    }, FULL_CKPT_PATH)
                    if verbose:
                        print(f"[train] saved checkpoint at epoch {ep+1} -> {CKPT_PATH} and {FULL_CKPT_PATH}", flush=True)
                except Exception as e:
                    if verbose:
                        print(f"[train] failed to save checkpoint at epoch {ep+1}: {e}", flush=True)
    except KeyboardInterrupt:
        if save_on_interrupt:
            try:
                torch.save(model.state_dict(), CKPT_PATH)
                torch.save({
                    "model_state": model.state_dict(),
                    "optimizer_state": opt.state_dict(),
                    "epoch": max(start_epoch, ep if 'ep' in locals() else 0),
                }, FULL_CKPT_PATH)
                print(f"[train] interrupted — checkpoint saved to {CKPT_PATH} and {FULL_CKPT_PATH}")
            except Exception as e:
                print(f"[train] interrupted — failed to save checkpoint: {e}")
        else:
            print("[train] interrupted — not saving (save_on_interrupt=False)")
        return

    # Final save
    try:
        torch.save(model.state_dict(), CKPT_PATH)
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "epoch": epochs,
        }, FULL_CKPT_PATH)
        print(f"Saved policy to {CKPT_PATH} and full checkpoint to {FULL_CKPT_PATH}")
    except Exception as e:
        print(f"[train] failed to save final checkpoint: {e}")


if __name__ == "__main__":
    train()
