
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from opacus.accountants import RDPAccountant  # type: ignore
except Exception:
    RDPAccountant = None


@dataclass
class LDPConfig:
    clip_norm: float = 1.0
    epsilon: float = 4.0
    delta: float = 1e-5
    noise_multiplier: float | None = None


def compute_noise_multiplier(ldp_cfg: LDPConfig) -> float:
    if ldp_cfg.noise_multiplier is not None:
        return float(ldp_cfg.noise_multiplier)
    if ldp_cfg.clip_norm <= 0:
        raise ValueError("clip_norm must be positive")
    if ldp_cfg.epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not (0 < ldp_cfg.delta < 1):
        raise ValueError("delta must be in (0, 1)")
    return float((2.0 * np.log(1.25 / ldp_cfg.delta)) ** 0.5 / ldp_cfg.epsilon)


def add_gaussian_noise_to_gradients(model: nn.Module, ldp_cfg: LDPConfig) -> None:
    noise_multiplier = compute_noise_multiplier(ldp_cfg)
    for param in model.parameters():
        if param.grad is None:
            continue
        noise = torch.normal(
            mean=0.0,
            std=float(noise_multiplier * ldp_cfg.clip_norm),
            size=param.grad.shape,
            device=param.grad.device,
            dtype=param.grad.dtype,
        )
        param.grad.add_(noise)


def train_one_client_with_ldp(
    model: nn.Module,
    dataloader,
    local_epochs: int,
    lr: float,
    ldp_cfg: LDPConfig,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], float, int]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    model.train()
    running_loss = 0.0
    num_batches = 0

    for _ in range(local_epochs):
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=ldp_cfg.clip_norm)
            add_gaussian_noise_to_gradients(model, ldp_cfg)

            optimizer.step()

            running_loss += float(loss.item())
            num_batches += 1

    avg_loss = running_loss / max(num_batches, 1)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return state, avg_loss, num_batches



def train_one_client_plain(
    model: nn.Module,
    dataloader,
    local_epochs: int,
    lr: float,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], float, int]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    model.train()
    running_loss = 0.0
    num_batches = 0

    for _ in range(local_epochs):
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            num_batches += 1

    avg_loss = running_loss / max(num_batches, 1)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return state, avg_loss, num_batches


def build_empty_privacy_report() -> dict:
    return {
        "engine": "none",
        "epsilon_estimate": 0.0,
        "target_epsilon": 0.0,
        "delta": 0.0,
        "clip_norm": 0.0,
        "noise_multiplier": 0.0,
        "num_steps": 0,
        "sample_rate": 0.0,
    }



def build_privacy_report(num_steps: int, sample_rate: float, ldp_cfg: LDPConfig) -> dict:
    noise_multiplier = compute_noise_multiplier(ldp_cfg)

    if RDPAccountant is not None:
        accountant = RDPAccountant()
        for _ in range(max(num_steps, 0)):
            accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
        epsilon = accountant.get_epsilon(delta=ldp_cfg.delta)
        engine = "opacus_rdp"
    else:
        epsilon = (num_steps * sample_rate) / max(noise_multiplier, 1e-12)
        engine = "analytic_fallback"

    return {
        "engine": engine,
        "epsilon_estimate": float(epsilon),
        "target_epsilon": float(ldp_cfg.epsilon),
        "delta": float(ldp_cfg.delta),
        "clip_norm": float(ldp_cfg.clip_norm),
        "noise_multiplier": float(noise_multiplier),
        "num_steps": int(num_steps),
        "sample_rate": float(sample_rate),
    }
