"""
CORAL Prototype: Decoupled Information Agent (IA) + Control Agent (CA)
- IA: Transformer-based encoder over recent states; outputs a message vector.
      Trained with (1) predictive loss (next-state prediction) and (2) Causal Influence Loss.
- CA: PPO actor-critic that consumes the IA's message to act.

"""
from __future__ import annotations
import math
import random
import time
from dataclasses import dataclass
from typing import Tuple, List, Dict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.amp import autocast, GradScaler
from collections import deque

# ----------------------------
# Utilities
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()
AMP_DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Global configuration variables (no hardcoded literals scattered)
# ----------------------------
ENTROPY_COEF = 0.02
# Optimizer learning rates
LR_CA = 7.5e-4
LR_IA = 1e-4
LOG_EPISODE_WINDOW = 50
EVAL_INTERVAL_STEPS = 10_000
EVAL_EPISODES = 5
DENSE_WARMUP_STEPS = 10_000
USE_INTRINSIC_BONUS = True
INTRINSIC_COEF = 0.05
INTRINSIC_DECAY = 0.0
OBS_NORM_ENABLE = True
OBS_NORM_EPS = 1e-8

# Message channel regularization and robustness (global variables)
MSG_DROPOUT_P = 0.1
MSG_NOISE_STD = 0.05
MSG_L2_COEF = 1e-5
MSG_DROPOUT_KEEP_EPS = 1e-6

# Threshold-focused shaping after warm-up
THRESHOLD_SHAPING_ENABLE = True
THRESHOLD_CENTER = 200
THRESHOLD_MARGIN = 20
THRESHOLD_BONUS = 0.1

# Evaluation ablations
EVAL_ABLATIONS = True
EVAL_NOISY_MSG_STD = 0.1

# Additional global schedules and stability configs
# Dense vs sparse reward mixing schedule (replaces hard warmup when enabled)
DENSE_MIX_ENABLE = True
DENSE_MIX_INIT = 1.0
DENSE_MIX_MIN = 0.05
DENSE_MIX_DECAY_STEPS = 50_000

# Entropy coefficient schedule
ENTROPY_SCHEDULE_ENABLE = True
ENTROPY_INIT = 0.05
ENTROPY_FINAL = 0.005
ENTROPY_DECAY_STEPS = 50_000

# Message dropout/noise schedules (used for robustness; acting uses clean message)
MSG_SCHEDULE_ENABLE = True
MSG_NOISE_INIT = 0.02
MSG_NOISE_FINAL = 0.0
MSG_NOISE_DECAY_STEPS = 20_000
MSG_DROPOUT_INIT = 0.05
MSG_DROPOUT_FINAL = 0.0
MSG_DROPOUT_DECAY_STEPS = 20_000

# PPO value function clipping
V_CLIP_EPS = 0.2

# IA update cadence (update every K rollouts)
IA_UPDATE_EVERY = 2

# PPO/Early stopping and optimization stabilities
EARLY_STOP_BY_KL = True
TARGET_KL = 0.035
MAX_GRAD_NORM = 1.0

# Global coefficients and epsilons (no hardcoded literals)
V_LOSS_COEF = 1.5
ADV_EPS = 1e-8
EV_EPS = 1e-8

# ---------------------------------
# EBT (Energy-Based Transformer) configs
# All defaults are global variables; features default OFF.
# ---------------------------------
EBT_ENABLE = True
EBT_TRAIN_ENABLE = True
EBT_INFER_ENABLE = True
LR_ENERGY = 5e-4
EBT_STEPS_ACT = 4
EBT_STEP_SIZE = 0.25
EBT_STOP_DELTA = 1e-4
EBT_KL_REG = 0.05
EBT_NEGATIVES = 4
EBT_MARGIN = 0.5
EBT_GRAD_PENALTY = 1e-4

class SparseCartPole(gym.Wrapper):
    """Sparse reward for CartPole: reward=+1 if you survive to max_steps, else 0.
    This isn't standard CartPole success, but good enough for a sparse signal.
    """
    def __init__(self, env, success_len: int = 200):
        super().__init__(env)
        self.success_len = success_len
        self.t = 0

    def reset(self, **kwargs):
        self.t = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        self.t += 1
        done = terminated or truncated
        reward = 0.0
        if done:
            reward = 1.0 if self.t >= self.success_len else 0.0
        # expose dense and sparse rewards for training choices
        info = dict(info) if info is not None else {}
        info['dense_r'] = r
        info['sparse_r'] = reward
        return obs, reward, terminated, truncated, info

# ----------------------------
# Running observation normalization
# ----------------------------
class RunningNorm:
    def __init__(self, shape: int, eps: float = OBS_NORM_EPS):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps
        self.eps = eps

    def update(self, x: np.ndarray):
        x = x.astype(np.float64)
        batch_mean = x
        batch_var = np.zeros_like(x)
        batch_count = 1.0
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / total_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * (self.count * batch_count / total_count)
        self.mean = new_mean
        self.var = M2 / max(total_count, self.eps)
        self.count = total_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / (np.sqrt(self.var) + self.eps)).astype(np.float32)

# ----------------------------
# Information Agent (IA)
# ----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:, :T]

class TransformerIA(nn.Module):
    def __init__(self, obs_dim: int, d_model: int = 64, nhead: int = 4, nlayers: int = 2,
                 msg_dim: int = 32):
        super().__init__()
        self.d_model = d_model
        self.obs_proj = nn.Linear(obs_dim, d_model)
        self.posenc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
                                                   dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.msg_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, msg_dim)
        )
        # Predictive head: predict next observation from the final hidden state
        self.pred_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, obs_dim)
        )

    def forward(self, obs_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        obs_seq: (B, T, obs_dim)
        Returns:
            msg: (B, msg_dim)
            pred_next_obs: (B, obs_dim)
            h_last: (B, d_model) (for introspection)
        """
        x = self.obs_proj(obs_seq)
        x = self.posenc(x)
        h = self.encoder(x)  # (B, T, d_model)
        h_last = h[:, -1, :]
        msg = self.msg_head(h_last)
        pred_next = self.pred_head(h_last)
        return msg, pred_next, h_last

# ----------------------------
# Control Agent (CA): PPO actor-critic consuming IA message
# ----------------------------
class ControlAgent(nn.Module):
    def __init__(self, msg_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.pi = nn.Sequential(
            nn.Linear(msg_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_actions)
        )
        self.v = nn.Sequential(
            nn.Linear(msg_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def policy(self, msg):
        logits = self.pi(msg)
        return Categorical(logits=logits)

    def value(self, msg):
        return self.v(msg).squeeze(-1)

# ----------------------------
# EBT Energy verifier head
# ----------------------------
class EnergyHead(nn.Module):
    def __init__(self, msg_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(msg_dim + n_actions, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, msg: torch.Tensor, act_probs: torch.Tensor) -> torch.Tensor:
        """
        msg: (B, D)
        act_probs: (B, A) probability simplex or one-hot
        returns: (B,) energy scalar (unnormalized; lower is better)
        """
        x = torch.cat([msg, act_probs], dim=-1)
        return self.net(x).squeeze(-1)

# ----------------------------
# Storage for on-policy rollouts
# ----------------------------
@dataclass
class StepStorage:
    obs: np.ndarray
    msg: np.ndarray
    action: int
    logp: float
    reward: float
    value: float
    done: bool
    next_obs: np.ndarray

class RolloutBuffer:
    def __init__(self):
        self.steps: List[StepStorage] = []

    def add(self, **kwargs):
        self.steps.append(StepStorage(**kwargs))

    def clear(self):
        self.steps.clear()

# ----------------------------
# CORAL Trainer
# ----------------------------
class CORAL:
    def __init__(self,
                 env_id: str = "CartPole-v1",
                 sparse: bool = True,
                 seq_len: int = 8,
                 d_model: int = 64,
                 msg_dim: int = 32,
                 lr_ca: float = LR_CA,
                 lr_ia: float = LR_IA,
                 gamma: float = 0.99,
                 lam: float = 0.95,
                 ppo_clip: float = 0.2,
                 ppo_epochs: int = 4,
                 batch_size: int = 64,
                 causal_coef: float = 1.0,
                 pred_coef: float = 1.0,
                 max_steps: int = 200_000,
                 entropy_coef: float = ENTROPY_COEF,
                 dense_warmup_steps: int = DENSE_WARMUP_STEPS,
                 use_intrinsic_bonus: bool = USE_INTRINSIC_BONUS,
                 intrinsic_coef: float = INTRINSIC_COEF,
                 intrinsic_decay: float = INTRINSIC_DECAY,
                 obs_norm_enable: bool = OBS_NORM_ENABLE,
                 eval_interval_steps: int = EVAL_INTERVAL_STEPS,
                 eval_episodes: int = EVAL_EPISODES,
                 log_window: int = LOG_EPISODE_WINDOW,
                 msg_dropout_p: float = MSG_DROPOUT_P,
                 msg_noise_std: float = MSG_NOISE_STD,
                 msg_l2_coef: float = MSG_L2_COEF,
                 threshold_shaping_enable: bool = THRESHOLD_SHAPING_ENABLE,
                 threshold_center: int = THRESHOLD_CENTER,
                 threshold_margin: int = THRESHOLD_MARGIN,
                 threshold_bonus: float = THRESHOLD_BONUS,
                 eval_ablations: bool = EVAL_ABLATIONS,
                 eval_noisy_msg_std: float = EVAL_NOISY_MSG_STD,
                 # New schedules and stability options
                 dense_mix_enable: bool = DENSE_MIX_ENABLE,
                 dense_mix_init: float = DENSE_MIX_INIT,
                 dense_mix_min: float = DENSE_MIX_MIN,
                 dense_mix_decay_steps: int = DENSE_MIX_DECAY_STEPS,
                 entropy_schedule_enable: bool = ENTROPY_SCHEDULE_ENABLE,
                 entropy_init: float = ENTROPY_INIT,
                 entropy_final: float = ENTROPY_FINAL,
                 entropy_decay_steps: int = ENTROPY_DECAY_STEPS,
                 msg_schedule_enable: bool = MSG_SCHEDULE_ENABLE,
                 msg_noise_init: float = MSG_NOISE_INIT,
                 msg_noise_final: float = MSG_NOISE_FINAL,
                 msg_noise_decay_steps: int = MSG_NOISE_DECAY_STEPS,
                 msg_dropout_init: float = MSG_DROPOUT_INIT,
                 msg_dropout_final: float = MSG_DROPOUT_FINAL,
                 msg_dropout_decay_steps: int = MSG_DROPOUT_DECAY_STEPS,
                 ia_update_every: int = IA_UPDATE_EVERY,
                 v_clip_eps: float = V_CLIP_EPS,
                 early_stop_by_kl: bool = EARLY_STOP_BY_KL,
                 target_kl: float = TARGET_KL,
                 max_grad_norm: float = MAX_GRAD_NORM,
                 # EBT params (fully global / no literals)
                 ebt_enable: bool = EBT_ENABLE,
                 ebt_train_enable: bool = EBT_TRAIN_ENABLE,
                 ebt_infer_enable: bool = EBT_INFER_ENABLE,
                 lr_energy: float = LR_ENERGY,
                 ebt_steps_act: int = EBT_STEPS_ACT,
                 ebt_step_size: float = EBT_STEP_SIZE,
                 ebt_stop_delta: float = EBT_STOP_DELTA,
                 ebt_kl_reg: float = EBT_KL_REG,
                 ebt_negatives: int = EBT_NEGATIVES,
                 ebt_margin: float = EBT_MARGIN,
                 ebt_grad_penalty: float = EBT_GRAD_PENALTY):
        base_env = gym.make(env_id)
        self.env = SparseCartPole(base_env) if sparse else base_env
        # Use a separate environment for evaluation to avoid interfering with training episode state
        base_eval_env = gym.make(env_id)
        self.eval_env = SparseCartPole(base_eval_env) if sparse else base_eval_env
        self.obs_dim = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        self.seq_len = seq_len
        self.gamma = gamma
        self.lam = lam
        self.ppo_clip = ppo_clip
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.causal_coef = causal_coef
        self.pred_coef = pred_coef
        self.max_steps = max_steps
        self.entropy_coef = entropy_coef
        self.dense_warmup_steps = dense_warmup_steps
        self.use_intrinsic_bonus = use_intrinsic_bonus
        self.intrinsic_coef = intrinsic_coef
        self.intrinsic_decay = intrinsic_decay
        self.obs_norm_enable = obs_norm_enable
        self.eval_interval_steps = eval_interval_steps
        self.eval_episodes = eval_episodes
        self.log_window = log_window
        # Message and shaping configs
        self.msg_dropout_p = msg_dropout_p
        self.msg_noise_std = msg_noise_std
        self.msg_l2_coef = msg_l2_coef
        self.threshold_shaping_enable = threshold_shaping_enable
        self.threshold_center = threshold_center
        self.threshold_margin = threshold_margin
        self.threshold_bonus = threshold_bonus
        self.eval_ablations = eval_ablations
        self.eval_noisy_msg_std = eval_noisy_msg_std
        # Schedules and stability
        self.dense_mix_enable = dense_mix_enable
        self.dense_mix_init = dense_mix_init
        self.dense_mix_min = dense_mix_min
        self.dense_mix_decay_steps = dense_mix_decay_steps
        self.entropy_schedule_enable = entropy_schedule_enable
        self.entropy_init = entropy_init
        self.entropy_final = entropy_final
        self.entropy_decay_steps = entropy_decay_steps
        self.msg_schedule_enable = msg_schedule_enable
        self.msg_noise_init = msg_noise_init
        self.msg_noise_final = msg_noise_final
        self.msg_noise_decay_steps = msg_noise_decay_steps
        self.msg_dropout_init = msg_dropout_init
        self.msg_dropout_final = msg_dropout_final
        self.msg_dropout_decay_steps = msg_dropout_decay_steps
        self.ia_update_every = ia_update_every
        self.v_clip_eps = v_clip_eps
        self.early_stop_by_kl = early_stop_by_kl
        self.target_kl = target_kl
        self.max_grad_norm = max_grad_norm
        # EBT config to attributes
        self.ebt_enable = bool(ebt_enable)
        self.ebt_train_enable = bool(ebt_train_enable)
        self.ebt_infer_enable = bool(ebt_infer_enable)
        self.ebt_steps_act = int(ebt_steps_act)
        self.ebt_step_size = float(ebt_step_size)
        self.ebt_stop_delta = float(ebt_stop_delta)
        self.ebt_kl_reg = float(ebt_kl_reg)
        self.ebt_negatives = int(ebt_negatives)
        self.ebt_margin = float(ebt_margin)
        self.ebt_grad_penalty = float(ebt_grad_penalty)
        # Align threshold center with env success length if present
        if hasattr(self.env, 'success_len'):
            self.threshold_center = int(getattr(self.env, 'success_len'))

        self.IA = TransformerIA(self.obs_dim, d_model=d_model, msg_dim=msg_dim).to(DEVICE)
        self.CA = ControlAgent(msg_dim, self.n_actions).to(DEVICE)
        # Energy verifier head (EBT)
        self.Energy = EnergyHead(msg_dim, self.n_actions).to(DEVICE)
        self.opt_ca = torch.optim.Adam(self.CA.parameters(), lr=lr_ca)
        self.opt_ia = torch.optim.Adam(self.IA.parameters(), lr=lr_ia)
        self.opt_energy = torch.optim.Adam(self.Energy.parameters(), lr=lr_energy)
        self.use_amp = USE_AMP
        self.scaler_ca = GradScaler(enabled=self.use_amp)
        self.scaler_ia = GradScaler(enabled=self.use_amp)
        self.scaler_energy = GradScaler(enabled=self.use_amp)

        # Log device info
        if DEVICE.type == "cuda":
            try:
                print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
            except Exception:
                print("Running on GPU")
        else:
            print("Running on CPU")

        self.buffer = RolloutBuffer()
        self.obs_hist: List[np.ndarray] = []
        self.obs_rms = RunningNorm(self.obs_dim) if self.obs_norm_enable else None
        self.ep_returns_window = deque(maxlen=self.log_window)
        self.ep_lens_window = deque(maxlen=self.log_window)
        self.ep_success_window = deque(maxlen=self.log_window)

    # ---------------
    # Helper methods
    # ---------------
    def _lin_sched(self, init: float, final: float, steps: int, t: int) -> float:
        """Linear schedule from init to final over 'steps' evaluated at time t."""
        steps = max(1, int(steps))
        frac = min(1.0, max(0.0, float(t) / float(steps)))
        return init + (final - init) * frac

    def _get_msg_and_pred(self, obs_hist_tensor: torch.Tensor):
        msg, pred_next, _ = self.IA(obs_hist_tensor)
        return msg, pred_next

    def _perturb_message(self, msg: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Apply dropout and Gaussian noise to IA message for robustness.
        Keeps expectation with inverted-dropout scaling.
        """
        if not training:
            return msg
        out = msg
        if self.msg_dropout_p > 0.0:
            keep_p = max(float(MSG_DROPOUT_KEEP_EPS), 1.0 - float(self.msg_dropout_p))
            mask = (torch.rand_like(out) < keep_p).to(out.dtype) / keep_p
            out = out * mask
        if self.msg_noise_std > 0.0:
            out = out + torch.randn_like(out) * float(self.msg_noise_std)
        return out

    def _ebt_refine_logits(self, msg: torch.Tensor, init_logits: torch.Tensor) -> torch.Tensor:
        """Refine action logits by minimizing energy with a KL tether to the base policy.
        - Only z (logits) receives gradients; IA/CA are treated as frozen during refinement.
        - Controlled by EBT_* global toggles.
        """
        if (not getattr(self, 'ebt_enable', False)) or (not getattr(self, 'ebt_infer_enable', False)):
            return init_logits
        steps = int(getattr(self, 'ebt_steps_act', 0))
        if steps <= 0:
            return init_logits
        step_size = float(getattr(self, 'ebt_step_size', 0.0))
        kl_reg = float(getattr(self, 'ebt_kl_reg', 0.0))
        stop_delta = float(getattr(self, 'ebt_stop_delta', 0.0))
        z = init_logits.detach().clone().requires_grad_(True)
        orig_probs = F.softmax(init_logits.detach(), dim=-1)
        prev_e = None
        for _ in range(steps):
            probs = F.softmax(z, dim=-1)
            e = self.Energy(msg.detach(), probs).mean()
            # KL(probs || orig_probs)
            kl = (probs * (probs.clamp(min=ADV_EPS).log() - orig_probs.clamp(min=ADV_EPS).log())).sum(dim=-1).mean()
            obj = e + kl_reg * kl
            if z.grad is not None:
                z.grad.zero_()
            obj.backward()
            with torch.no_grad():
                z -= step_size * z.grad
            if prev_e is not None and abs((prev_e - e).item()) < stop_delta:
                break
            prev_e = e.detach()
        return z.detach()

    def _compute_advantages(self, rewards, values, dones, last_value=0.0):
        adv = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * last_value * mask - values[t]
            gae = delta + self.gamma * self.lam * mask * gae
            adv.insert(0, gae)
            last_value = values[t]
        returns = [a + v for a, v in zip(adv, values)]
        adv = np.array(adv, dtype=np.float32)
        returns = np.array(returns, dtype=np.float32)
        adv = (adv - adv.mean()) / (adv.std() + ADV_EPS)
        return adv, returns

    # -----------------------------
    # Causal Influence Loss (CIL)
    # -----------------------------
    def causal_influence_loss(self, msgs: torch.Tensor, base_msgs: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
        """Utility-weighted KL divergence between CA policies with and without the IA message.
        - msgs: current IA message (grad flows to IA only)
        - base_msgs: a counterfactual message (no grad) — e.g., zeros or detached
        - advantages: standardized advantages (positive => helpful to increase influence)

        We treat CA as fixed for this term (stop-grad) to target IA's output.
        """
        with torch.no_grad():
            pi_base = self.CA.policy(base_msgs)
        pi = self.CA.policy(msgs)  # gradients will NOT be used for CA during IA step
        # KL(pi || pi_base) = sum p * (log p - log q)
        kl = torch.distributions.kl.kl_divergence(pi, pi_base)
        # Weight by (positive) advantage; encourage influence when it correlates with utility
        weight = advantages.clamp(min=0.0)
        loss = -(weight * kl).mean()  # maximize KL under positive advantage
        return loss

    # -----------------------------
    # Training loop
    # -----------------------------
    def train(self, total_env_steps: int = 50_000, rollout_horizon: int = 1024):
        obs, _ = self.env.reset(seed=42)
        if self.obs_rms is not None:
            self.obs_rms.update(obs)
            n_obs = self.obs_rms.normalize(obs)
        else:
            n_obs = obs
        self.obs_hist = [np.zeros_like(n_obs) for _ in range(self.seq_len-1)] + [n_obs]
        ep_dense_return = 0.0
        ep_sparse_return = 0.0
        ep_len = 0
        threshold_bonus_given = False
        step_count = 0
        episodes = 0
        next_eval_step = self.eval_interval_steps
        update_iter = 0

        while step_count < total_env_steps:
            # Collect rollout
            self.buffer.clear()
            # Update schedules (message noise/dropout) per-iteration
            if self.msg_schedule_enable:
                self.msg_noise_std = self._lin_sched(self.msg_noise_init, self.msg_noise_final, self.msg_noise_decay_steps, step_count)
                self.msg_dropout_p = self._lin_sched(self.msg_dropout_init, self.msg_dropout_final, self.msg_dropout_decay_steps, step_count)
            for _ in range(rollout_horizon):
                obs_seq = np.stack(self.obs_hist[-self.seq_len:], axis=0)  # (T, obs_dim)
                obs_seq_t = torch.tensor(obs_seq, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                with torch.no_grad():
                    msg, pred_next, _ = self.IA(obs_seq_t)
                    base_logits = self.CA.pi(msg)
                # Optional EBT refinement of logits (no grads to IA/CA)
                refined_logits = self._ebt_refine_logits(msg.detach(), base_logits.detach())
                with torch.no_grad():
                    dist = Categorical(logits=refined_logits)
                    value = self.CA.value(msg)
                    action = dist.sample()
                    logp = dist.log_prob(action)
                next_obs, reward, terminated, truncated, info = self.env.step(action.item())
                done = terminated or truncated

                # Reward selection: dense/sparse mixing schedule (or legacy warmup)
                base_reward = info.get('dense_r', reward)
                if self.dense_mix_enable:
                    w = max(float(self.dense_mix_min), self._lin_sched(self.dense_mix_init, 0.0, self.dense_mix_decay_steps, step_count))
                    used_reward = float(w * base_reward + (1.0 - w) * reward)
                    use_dense_now = False  # for logging below
                else:
                    use_dense_now = (self.dense_warmup_steps > 0) and (step_count < self.dense_warmup_steps)
                    used_reward = float(base_reward if use_dense_now else reward)

                # Threshold-focused shaping (one-time per episode) near success boundary
                if (not use_dense_now) and self.threshold_shaping_enable and (not threshold_bonus_given):
                    if abs(ep_len - int(self.threshold_center)) <= int(self.threshold_margin):
                        used_reward += float(self.threshold_bonus)
                        threshold_bonus_given = True

                # Normalize next obs
                if self.obs_rms is not None:
                    self.obs_rms.update(next_obs)
                    n_next_obs = self.obs_rms.normalize(next_obs)
                else:
                    n_next_obs = next_obs

                # Intrinsic bonus from prediction error (detach to avoid IA gradients via reward)
                if self.use_intrinsic_bonus and self.intrinsic_coef != 0.0:
                    with torch.no_grad():
                        target_next_t = torch.tensor(n_next_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                        pred_err = F.mse_loss(pred_next, target_next_t).item()
                    coef = max(0.0, self.intrinsic_coef - self.intrinsic_decay * step_count)
                    used_reward += coef * pred_err

                self.buffer.add(
                    obs=obs_seq.copy(),
                    msg=msg.squeeze(0).cpu().numpy(),
                    action=int(action.item()),
                    logp=float(logp.item()),
                    reward=float(used_reward),
                    value=float(value.item()),
                    done=bool(done),
                    next_obs=n_next_obs,
                )

                # Track separate dense and sparse episode returns for visibility
                ep_dense_return += float(base_reward)
                ep_sparse_return += float(reward)
                ep_len += 1
                step_count += 1

                # Update history
                self.obs_hist.append(n_next_obs)
                obs = next_obs

                if done:
                    episodes += 1
                    # record success under sparse metric (reward at termination)
                    success = 1.0 if reward >= 0.5 else 0.0
                    self.ep_returns_window.append(ep_sparse_return)
                    self.ep_lens_window.append(ep_len)
                    self.ep_success_window.append(success)

                    obs, _ = self.env.reset()
                    if self.obs_rms is not None:
                        self.obs_rms.update(obs)
                        n_obs = self.obs_rms.normalize(obs)
                    else:
                        n_obs = obs
                    self.obs_hist = [np.zeros_like(n_obs) for _ in range(self.seq_len-1)] + [n_obs]
                    avg_ret = np.mean(self.ep_returns_window) if len(self.ep_returns_window) else 0.0
                    avg_len = np.mean(self.ep_lens_window) if len(self.ep_lens_window) else 0.0
                    succ = np.mean(self.ep_success_window) if len(self.ep_success_window) else 0.0
                    print(f"Episode {episodes} | dense_ep_ret={ep_dense_return:.1f} | sparse_ep_ret={ep_sparse_return:.1f} | len={ep_len} | avg_sparse_ret={avg_ret:.2f} | avg_len={avg_len:.1f} | success_rate={succ:.2f}")
                    ep_dense_return, ep_sparse_return, ep_len = 0.0, 0.0, 0
                    threshold_bonus_given = False

                if step_count >= total_env_steps:
                    break

            # Prepare batch tensors
            obs_batch = torch.tensor(np.stack([s.obs for s in self.buffer.steps], axis=0), dtype=torch.float32, device=DEVICE)  # (N, T, obs)
            msg_batch = torch.tensor(np.stack([s.msg for s in self.buffer.steps], axis=0), dtype=torch.float32, device=DEVICE)
            actions = torch.tensor([s.action for s in self.buffer.steps], dtype=torch.long, device=DEVICE)
            old_logp = torch.tensor([s.logp for s in self.buffer.steps], dtype=torch.float32, device=DEVICE)
            rewards = np.array([s.reward for s in self.buffer.steps], dtype=np.float32)
            values = np.array([s.value for s in self.buffer.steps], dtype=np.float32)
            dones = np.array([s.done for s in self.buffer.steps], dtype=np.bool_)
            old_values = torch.tensor(values, dtype=torch.float32, device=DEVICE)
            # Bootstrap with value of last state if rollout ended without a terminal
            if len(self.buffer.steps) > 0 and not self.buffer.steps[-1].done:
                # Use current obs history to compute last state value
                last_obs_seq = np.stack(self.obs_hist[-self.seq_len:], axis=0)
                last_obs_seq_t = torch.tensor(last_obs_seq, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                with torch.no_grad():
                    last_msg, _, _ = self.IA(last_obs_seq_t)
                    last_value = float(self.CA.value(last_msg).item())
            else:
                last_value = 0.0
            adv, rets = self._compute_advantages(rewards, values, dones, last_value)
            advantages = torch.tensor(adv, dtype=torch.float32, device=DEVICE)
            returns = torch.tensor(rets, dtype=torch.float32, device=DEVICE)

            # -----------------
            # PPO update (CA)
            # -----------------
            pg_losses = []
            v_losses = []
            entropies = []
            approx_kls = []
            clipfracs = []
            ratio_means = []
            early_stopped = False
            # Entropy schedule per-update
            if self.entropy_schedule_enable:
                cur_entropy_coef = self._lin_sched(self.entropy_init, self.entropy_final, self.entropy_decay_steps, step_count)
            else:
                cur_entropy_coef = self.entropy_coef
            # Explained variance of baseline value vs returns (pre-update)
            with torch.no_grad():
                rets_np = returns.detach().cpu().numpy()
                vals_np = old_values.detach().cpu().numpy()
                ev = 1.0 - (np.var(rets_np - vals_np) / (np.var(rets_np) + EV_EPS))
            for _ in range(self.ppo_epochs):
                idx = np.random.permutation(len(actions))
                for start in range(0, len(actions), self.batch_size):
                    batch_idx = idx[start:start+self.batch_size]
                    b_msgs = msg_batch[batch_idx].detach()  # detach so CA learns on fixed messages
                    b_actions = actions[batch_idx]
                    b_old_logp = old_logp[batch_idx]
                    b_adv = advantages[batch_idx]
                    b_returns = returns[batch_idx]
                    b_values = old_values[batch_idx]

                    with autocast(device_type=AMP_DEVICE_TYPE, enabled=self.use_amp):
                        dist = self.CA.policy(b_msgs)
                        logp = dist.log_prob(b_actions)
                        ratio = torch.exp(logp - b_old_logp)
                        clipped = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * b_adv
                        pg_loss = -(torch.min(ratio * b_adv, clipped)).mean()
                        v_pred = self.CA.value(b_msgs)
                        # PPO value clipping
                        v_pred_clipped = b_values + (v_pred - b_values).clamp(-self.v_clip_eps, self.v_clip_eps)
                        v_loss_unclipped = F.mse_loss(v_pred, b_returns)
                        v_loss_clipped = F.mse_loss(v_pred_clipped, b_returns)
                        v_loss = torch.max(v_loss_unclipped, v_loss_clipped)
                        ent = dist.entropy().mean()
                        loss = pg_loss + V_LOSS_COEF * v_loss - cur_entropy_coef * ent

                    self.opt_ca.zero_grad()
                    if self.use_amp:
                        self.scaler_ca.scale(loss).backward()
                        self.scaler_ca.unscale_(self.opt_ca)
                        torch.nn.utils.clip_grad_norm_(self.CA.parameters(), self.max_grad_norm)
                        self.scaler_ca.step(self.opt_ca)
                        self.scaler_ca.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.CA.parameters(), self.max_grad_norm)
                        self.opt_ca.step()

                    with torch.no_grad():
                        approx_kl_val = (b_old_logp - logp).mean().item()
                        approx_kls.append(approx_kl_val)
                        pg_losses.append(pg_loss.item())
                        v_losses.append(v_loss.item())
                        entropies.append(ent.item())
                        clipfracs.append((torch.abs(ratio - 1.0) > self.ppo_clip).float().mean().item())
                        ratio_means.append(ratio.mean().item())
                        if self.early_stop_by_kl and approx_kl_val > self.target_kl:
                            early_stopped = True
                            break
                if early_stopped:
                    break

            # -----------------
            # IA update (predictive + causal influence)
            # -----------------
            if (self.ia_update_every <= 1) or ((update_iter % self.ia_update_every) == 0):
                # Predictive loss: next-obs prediction conditioned on sequence
                # Freeze CA params during IA step so gradients target IA only
                for p in self.CA.parameters():
                    p.requires_grad_(False)
                with autocast(device_type=AMP_DEVICE_TYPE, enabled=self.use_amp):
                    pred_msgs, pred_next_obs, _ = self.IA(obs_batch)
                    true_next_obs = torch.tensor(np.stack([s.next_obs for s in self.buffer.steps], axis=0),
                                                 dtype=torch.float32, device=DEVICE)
                    pred_loss = F.mse_loss(pred_next_obs, true_next_obs)

                    # Causal influence loss comparing with a counterfactual baseline (zero message)
                    base_msgs = torch.zeros_like(pred_msgs)
                    cil = self.causal_influence_loss(pred_msgs, base_msgs, advantages)

                    # Message L2 penalty to encourage concise messages
                    msg_l2 = pred_msgs.pow(2).mean()

                    ia_loss = self.pred_coef * pred_loss + self.causal_coef * cil + self.msg_l2_coef * msg_l2

                self.opt_ia.zero_grad()
                if self.use_amp:
                    self.scaler_ia.scale(ia_loss).backward()
                    self.scaler_ia.unscale_(self.opt_ia)
                    torch.nn.utils.clip_grad_norm_(self.IA.parameters(), self.max_grad_norm)
                    self.scaler_ia.step(self.opt_ia)
                    self.scaler_ia.update()
                else:
                    ia_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.IA.parameters(), self.max_grad_norm)
                    self.opt_ia.step()
                # Unfreeze CA
                for p in self.CA.parameters():
                    p.requires_grad_(True)

                # -----------------
                # EBT energy head update (ranking + gradient penalty)
                # -----------------
                if self.ebt_enable and self.ebt_train_enable:
                    with autocast(device_type=AMP_DEVICE_TYPE, enabled=self.use_amp):
                        # Positive pairs: (msg, taken_action)
                        pos_onehot = F.one_hot(actions, num_classes=self.n_actions).float()
                        # Use a detached clone that requires grad for GP without affecting IA
                        pred_msgs_gp = pred_msgs.detach().clone().requires_grad_(True)
                        e_pos = self.Energy(pred_msgs_gp, pos_onehot)
                        # Negative actions per sample (exclude the taken action)
                        N = actions.size(0)
                        K = max(1, int(self.ebt_negatives))
                        a = actions.unsqueeze(1).expand(N, K)
                        neg = torch.randint(low=0, high=self.n_actions - 1, size=(N, K), device=DEVICE)
                        neg = neg + (neg >= a).long()
                        neg_onehot = F.one_hot(neg, num_classes=self.n_actions).float().view(N * K, self.n_actions)
                        msgs_tiled = pred_msgs.detach().unsqueeze(1).expand(N, K, pred_msgs.size(-1)).contiguous().view(N * K, pred_msgs.size(-1))
                        e_neg = self.Energy(msgs_tiled, neg_onehot).view(N, K)
                        # Margin ranking: max(0, margin + e_pos - e_neg)
                        hinge = F.relu(float(self.ebt_margin) + e_pos.unsqueeze(1) - e_neg)
                        rank_loss = hinge.mean()
                        # Gradient penalty wrt message (smooth energy landscape)
                        gp = 0.0
                        try:
                            grad_msg = torch.autograd.grad(e_pos.sum(), pred_msgs_gp, create_graph=True, retain_graph=True)[0]
                            gp = (grad_msg.pow(2).mean())
                        except Exception:
                            gp = torch.tensor(0.0, device=DEVICE)
                        energy_loss = rank_loss + float(self.ebt_grad_penalty) * gp
                    self.opt_energy.zero_grad()
                    if self.use_amp:
                        self.scaler_energy.scale(energy_loss).backward()
                        self.scaler_energy.unscale_(self.opt_energy)
                        torch.nn.utils.clip_grad_norm_(self.Energy.parameters(), self.max_grad_norm)
                        self.scaler_energy.step(self.opt_energy)
                        self.scaler_energy.update()
                    else:
                        energy_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.Energy.parameters(), self.max_grad_norm)
                        self.opt_energy.step()
            else:
                # Skip IA update this iteration
                pred_loss = torch.tensor(0.0)
                cil = torch.tensor(0.0)
                energy_loss = torch.tensor(0.0)

            # Raw (unweighted) KL for IA influence interpretability (ensure dtype matches CA params)
            with torch.no_grad():
                ca_dtype = next(self.CA.parameters()).dtype
                # Use current batch messages for interpretability if available
                if 'pred_msgs' in locals():
                    pred_msgs_f = pred_msgs.detach().to(dtype=ca_dtype)
                    pi = self.CA.policy(pred_msgs_f)
                    pi_base = self.CA.policy(torch.zeros_like(pred_msgs_f))
                    raw_kl = torch.distributions.kl.kl_divergence(pi, pi_base).mean().item()
                else:
                    raw_kl = 0.0

            # Logging line with richer stats
            print(
                f"Update | steps={step_count} | pred_loss={pred_loss.item():.4f} | cil={cil.item():.4f} | raw_kl={raw_kl:.4f} | "
                f"energy_loss={(energy_loss.item() if 'energy_loss' in locals() else 0.0):.4f} | "
                f"msg_l2={(pred_msgs.pow(2).mean().item() if 'pred_msgs' in locals() else 0.0):.4f} | pg={np.mean(pg_losses):.4f} | v={np.mean(v_losses):.4f} | "
                f"ent={np.mean(entropies):.3f} | kl~={np.mean(approx_kls):.4f} | clipfrac={np.mean(clipfracs) if clipfracs else 0.0:.3f} | "
                f"ratio={np.mean(ratio_means) if ratio_means else 1.0:.3f} | ev={ev:.3f} | kl_stop={'Y' if early_stopped else 'N'}"
            )

            update_iter += 1

            # Periodic evaluation
            if self.eval_interval_steps > 0 and step_count >= next_eval_step:
                def run_eval(mode: str):
                    rets, lens = [], []
                    for _ in range(self.eval_episodes):
                        e_obs, _ = self.eval_env.reset()
                        n_e_obs = self.obs_rms.normalize(e_obs) if self.obs_rms is not None else e_obs
                        e_hist = [np.zeros_like(n_e_obs) for _ in range(self.seq_len-1)] + [n_e_obs]
                        e_ret = 0.0
                        e_len = 0
                        while True:
                            e_seq = np.stack(e_hist[-self.seq_len:], axis=0)
                            e_seq_t = torch.tensor(e_seq, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                            with torch.no_grad():
                                e_msg, _, _ = self.IA(e_seq_t)
                                if mode == 'zero':
                                    e_in = torch.zeros_like(e_msg)
                                elif mode == 'noisy':
                                    e_in = e_msg + torch.randn_like(e_msg) * float(self.eval_noisy_msg_std)
                                else:
                                    e_in = e_msg
                                base_logits = self.CA.pi(e_in)
                            ref_logits = self._ebt_refine_logits(e_in.detach(), base_logits.detach())
                            with torch.no_grad():
                                e_dist = Categorical(logits=ref_logits)
                                e_action = torch.argmax(e_dist.probs, dim=-1)
                            e_next_obs, e_reward, e_term, e_trunc, _e_info = self.eval_env.step(e_action.item())
                            n_e_next = self.obs_rms.normalize(e_next_obs) if self.obs_rms is not None else e_next_obs
                            e_hist.append(n_e_next)
                            e_ret += e_reward
                            e_len += 1
                            if e_term or e_trunc:
                                break
                        rets.append(e_ret)
                        lens.append(e_len)
                    return float(np.mean(rets)), float(np.mean(lens))

                if self.eval_ablations:
                    avg_ret_n, avg_len_n = run_eval('normal')
                    avg_ret_z, avg_len_z = run_eval('zero')
                    avg_ret_x, avg_len_x = run_eval('noisy')
                    print(
                        f"Eval | steps={step_count} | normal_ret={avg_ret_n:.2f} len={avg_len_n:.1f} | "
                        f"zero_ret={avg_ret_z:.2f} len={avg_len_z:.1f} | noisy_ret={avg_ret_x:.2f} len={avg_len_x:.1f}"
                    )
                else:
                    avg_ret, avg_len = run_eval('normal')
                    print(f"Eval | steps={step_count} | avg_return={avg_ret:.2f} | avg_len={avg_len:.1f}")
                next_eval_step += self.eval_interval_steps

        print("Training complete.")


if __name__ == "__main__":
    cfg = dict(
        env_id="CartPole-v1",
        sparse=True,
        seq_len=8,
        d_model=64,
        msg_dim=32,
        lr_ca=LR_CA,
        lr_ia=LR_IA,
        gamma=0.99,
        lam=0.95,
        ppo_clip=0.2,
        ppo_epochs=4,
        batch_size=64,
        causal_coef=1.0,
        pred_coef=1.0,
        max_steps=50_000,
        entropy_coef=ENTROPY_COEF,
        dense_warmup_steps=DENSE_WARMUP_STEPS,
        use_intrinsic_bonus=USE_INTRINSIC_BONUS,
        intrinsic_coef=INTRINSIC_COEF,
        intrinsic_decay=INTRINSIC_DECAY,
        obs_norm_enable=OBS_NORM_ENABLE,
        eval_interval_steps=EVAL_INTERVAL_STEPS,
        eval_episodes=EVAL_EPISODES,
        log_window=LOG_EPISODE_WINDOW,
    )
    coral = CORAL(**cfg)
    coral.train(total_env_steps=20_000, rollout_horizon=1024)
