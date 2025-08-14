# ARC-AGI-2 + Program-Synthesis Agent

This repository contains:
- The ARC-AGI-2 dataset (public training and evaluation tasks).
- A config-driven program-synthesis system for ARC-style grid transformations:
  - A compact DSL with deterministic, testable ops.
  - Enumerative and beam search over token programs.
  - Optional incremental typed beam search guided by a policy model.

All parameters (tokens, constants, search weights/knobs) are controlled globally via config. No literals are hardcoded in code.


## Directory structure

- `app/`
  - `config.py`: Centralized global config loader/merger from `state/config.yaml`. Exposes constants and maps.
  - `dsl/`
    - `ops.py`: Deterministic NumPy grid ops (primitive DSL operations).
    - `program.py`: Token/AST representation, arity, parsing/validation.
    - `interpreter.py`: Executes token programs on input grids; pair-consistency utilities.
  - `search/`
    - `enumerator.py`: Typed arg enumeration for ops (deterministic arg spaces).
    - `beam_search.py`: Classic beam search and incremental typed beam search (policy-guided) with execution pruning.
  - `eval/`
    - `arc_eval.py`: Evaluation loop over ARC tasks with/without policy.
  - `utils/`
    - `tokens.py`: Token maps, vocabulary, dynamic tokens for numbers/colors/rots/reflects, and `OP_ARITY`.
- `data/`
  - `training/` (1000 tasks), `evaluation/` (120 tasks).
- `state/`
  - `config.yaml`: Project configuration (the single source of truth for knobs and enabled ops).
  - `policy.pt` (optional): Policy checkpoint for incremental typed beam search.
- `requirements.txt`


## DSL overview

The DSL operates on integer grid images (NumPy arrays). Each op is minimal, deterministic, and testable. All constants (e.g., number of colors, connectivity, background color) come from config.

Existing primitives include `MAP_COLOR`, `FILTER_EQ`, `ROTATE`, `REFLECT`, `CROP`, `PAD`, plus the binary `COMPOSE` operator.

### Newly added ops (config-gated)
All new ops are available only when listed in `state/config.yaml` under `dsl.enable_ops`. Arity is defined in `app/utils/tokens.py` (`OP_ARITY`). Implementations live in `app/dsl/ops.py`; interpreter wiring is in `app/dsl/interpreter.py`.

- `FIND_COMPONENTS(offset: num)`
  - Relabels connected components of non-background cells using `(offset + 1 + i) % NUM_COLORS`.
  - Connectivity set by `MODEL_CFG.components_connectivity` (4 or 8).
- `GET_BBOX(color: col)`
  - Crops grid to tight bounding box of `color`. If absent, returns input unchanged.
- `PAINT_OBJECT(src_color: col, dst_color: col)`
  - Paints all `src_color` cells to `dst_color`.
- `COUNT_COLOR(color: col)`
  - Counts occurrences of `color`; returns a constant grid filled with `(count % NUM_COLORS)`.
- `MAJORITY_COLOR(tie_color: col)`
  - Returns a constant grid of the most frequent color; ties resolved to `tie_color`.
- `TRANSLATE(dy: num, dx: num)`
  - Shifts grid down/right by `(dy, dx)`; out-of-bounds filled with background.
- `DRAW_LINE(y0: num, x0: num, y1: num, x1: num, color: col)`
  - Bresenham-like line drawing; coordinates clamped to grid.
- `FILL_RECT(top: num, left: num, height: num, width: num, color: col)`
  - Fills axis-aligned rectangle; bounds clamped.
- `REPEAT_TILE(ny: num, nx: num)`
  - Tiles the grid `(ny, nx)` and crops to `DSL_MAX_GRID_SIZE`.
- `OVERLAY_UNION(color_a: col, color_b: col, out_color: col)`
  - Writes `out_color` where grid equals `color_a` or `color_b`.
- `OVERLAY_INTERSECT(color_a: col, color_b: col, out_color: col)`
  - Writes `out_color` at `color_a` cells adjacent to any `color_b` cell;
    adjacency respects `components_connectivity` (4/8).


## Config system

- Primary file: `state/config.yaml` (merged into Python via `app/config.py`).
- Areas:
  - `dsl.*`: `enable_ops`, depth, numbers/colors/rots/reflects, grid size.
  - `tokens.*`: Token IDs for ops and dynamic token ranges.
  - `search.*`: Beam width, max nodes, time budgets, incremental typed knobs.
  - `infer.*`: Evaluation/inference options, workers, use_policy toggle.
  - `model.*`: Component connectivity, background color, policy settings.

All DSL token IDs and op enablement are controlled here for ablations. Search weights/knobs are fully exposed—no literals are hardcoded.


## Search

- `enumerator.py` emits typed argument choices for each enabled op.
- `beam_search.py` implements:
  - Classic beam over enumerator sequences.
  - Incremental typed beam search:
    - Tracks expected token kinds and masks policy logits by type.
    - Execution-guided pruning with heuristic loss on training pairs.
    - Scores combine policy log-probs and execution heuristics (weights from config).


## Interpreter and programs

- `program.py` defines token/AST conversions and validity checks, using centralized `OP_ARITY` from `tokens.py`.
- `interpreter.py` executes ASTs deterministically on NumPy grids. Pair consistency check utilities are provided.


## Evaluation

File: `app/eval/arc_eval.py`.

- Enumerative beam (no policy):
  ```bash
  # Evaluate all tasks (single-process)
  python -m app.eval.arc_eval

  # Tiny eval of N tasks
  python -c "from app.eval.arc_eval import evaluate; evaluate(2)"
  ```
- Incremental typed beam (policy-guided):
  - Place a compatible checkpoint at `state/policy.pt` and set `infer.use_policy: true` in `state/config.yaml`.
  - Optional: adjust `search.policy_beam_width`, `search.policy_max_nodes`.


## Training (policy)

A `ProgramPolicy` model (not required for enumerative search) can be trained to guide incremental typed search.
- Define/inspect the model in `app/model/program_policy.py` (if present in your branch).
- Ensure the policy’s vocabulary size matches the current token set in `app/utils/tokens.py`.
- Save checkpoints as `state/policy.pt` for evaluation.


## Reproducibility & conventions

- All constants come from `config.py` merged with `state/config.yaml`.
- No hardcoded numerical literals in code; use global variables from config only.
- Deterministic NumPy ops; seeded randomness (if any) must be drawn from config.


## Quick smoke tests

- Import and basic op execution:
  ```bash
  python -c "import numpy as np; from app.dsl import ops as OP; g=np.zeros((3,3),int); OP.translate(g,1,1); print('OK')"
  ```
- Tiny enumerative eval on 2 tasks:
  ```bash
  python -c "from app.eval.arc_eval import evaluate; evaluate(2)"
  ```
- Tiny incremental typed eval (requires `state/policy.pt` or a dummy policy):
  ```bash
  # Real policy
  python -m app.eval.arc_eval
  ```


## What changed in this fork

- Added minimal deterministic DSL ops:
  - `TRANSLATE`, `DRAW_LINE`, `FILL_RECT`, `REPEAT_TILE`,
    `GET_BBOX`, `PAINT_OBJECT`, `FIND_COMPONENTS`,
    `COUNT_COLOR`, `MAJORITY_COLOR`, `OVERLAY_UNION`, `OVERLAY_INTERSECT`.
- Extended `OP_ARITY` and typed argument spaces (`tokens.py`, `enumerator.py`, `beam_search.py`).
- Integrated ops into interpreter (`interpreter.py`).
- Implemented incremental typed beam search with execution-guided pruning.
- Exposed all search and DSL knobs in `state/config.yaml` (no hardcoded values).


## Limitations & next steps

- `TRANSLATE` currently clamps to non-negative shifts (down/right) for minimality.
- Some ops (e.g., `GET_BBOX`) can change grid shape; ensure downstream code handles shape changes.
- Consider unit tests per op for regression safety.
- If resuming from older checkpoints, ensure token vocab alignment or retrain.


## License

Refer to the original ARC-AGI-2 dataset license and any licenses for added code where applicable.
