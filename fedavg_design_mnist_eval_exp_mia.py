
from __future__ import annotations

import argparse
import base64
import csv
import json
import socket
import struct
import sys
import time
import unittest
from dataclasses import asdict, dataclass
from multiprocessing import Barrier, Event, Process
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from ldp_module_eval import (
    LDPConfig,
    build_no_privacy_report,
    build_privacy_report,
    train_one_client_plain,
    train_one_client_with_ldp,
)
from he_module_eval import (
    HEConfig,
    aggregate_encrypted_payloads,
    build_he_context_blob,
    decrypt_aggregated_payload,
    encrypt_state_dict,
)

try:
    from torchvision import datasets, transforms
except Exception:
    datasets = None
    transforms = None


@dataclass
class FLConfig:
    host: str = "127.0.0.1"
    port: int = 50051
    num_clients: int = 3
    rounds: int = 5
    local_epochs: int = 1
    batch_size: int = 32
    lr: float = 0.01
    input_dim: int = 20
    num_classes: int = 2
    samples_per_client: int = 256
    seed: int = 42
    socket_timeout: float = 20.0

    dataset: str = "mnist"
    data_root: str = "./data"
    download_data: bool = True
    # For membership inference attack experiments:
    # limit the number of MNIST training samples used by FL so that a
    # same-distribution train holdout can be used as non-member samples.
    # None or <=0 means use all 60,000 MNIST training samples.
    mnist_train_size: int | None = None

    ldp_clip_norm: float = 1.0
    ldp_epsilon: float = 4.0
    ldp_delta: float = 1e-5
    ldp_noise_multiplier: float | None = None
    disable_ldp: bool = False

    # he_mode:
    #   "real" -> 使用 TenSEAL/BFV 真实同态加密路径；
    #   "mock" -> 使用 Mock HE 路径，便于快速验证流程；
    #   "none" -> 不使用 HE，采用明文等权聚合，用作无保护 FedAvg baseline。
    he_mode: str = "real"

    he_scale: float = 1000.0
    he_poly_modulus_degree: int = 8192
    he_plain_modulus: int = 1032193

    results_dir: str = "./results"
    experiment_name: str = "default_run"
    save_plots: bool = True


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SyntheticClientDataset(Dataset):
    def __init__(self, client_id: int, n_samples: int, input_dim: int, num_classes: int, seed: int):
        super().__init__()
        rng = np.random.default_rng(seed + 1000 * client_id)
        x = rng.normal(size=(n_samples, input_dim)).astype(np.float32)
        w = rng.normal(size=(input_dim, num_classes)).astype(np.float32)
        logits = x @ w + 0.25 * client_id
        y = logits.argmax(axis=1).astype(np.int64)
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def flatten_tensor(x):
    return x.view(-1)


def mnist_transform():
    if transforms is None:
        raise ImportError("torchvision is required for MNIST experiments")
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(flatten_tensor),
    ])


def prepare_data(cfg: FLConfig) -> None:
    if cfg.dataset.lower() != "mnist":
        return
    if datasets is None:
        raise ImportError("torchvision is required for MNIST experiments")
    datasets.MNIST(root=cfg.data_root, train=True, download=cfg.download_data, transform=mnist_transform())
    datasets.MNIST(root=cfg.data_root, train=False, download=cfg.download_data, transform=mnist_transform())


def build_train_dataset(cfg: FLConfig, client_id: int) -> Dataset:
    if cfg.dataset.lower() == "mnist":
        if datasets is None:
            raise ImportError("torchvision is required for MNIST experiments")
        return datasets.MNIST(root=cfg.data_root, train=True, download=False, transform=mnist_transform())
    return SyntheticClientDataset(
        client_id=client_id,
        n_samples=cfg.samples_per_client,
        input_dim=cfg.input_dim,
        num_classes=cfg.num_classes,
        seed=cfg.seed,
    )


def build_test_dataset(cfg: FLConfig) -> Dataset:
    if cfg.dataset.lower() == "mnist":
        if datasets is None:
            raise ImportError("torchvision is required for MNIST experiments")
        return datasets.MNIST(root=cfg.data_root, train=False, download=False, transform=mnist_transform())
    return SyntheticClientDataset(
        client_id=9999,
        n_samples=max(cfg.samples_per_client, 512),
        input_dim=cfg.input_dim,
        num_classes=cfg.num_classes,
        seed=cfg.seed + 9999,
    )


def split_indices_iid(dataset_size: int, num_clients: int, seed: int) -> List[List[int]]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()
    split_size = dataset_size // num_clients
    client_indices: List[List[int]] = []
    for i in range(num_clients):
        start = i * split_size
        end = dataset_size if i == num_clients - 1 else (i + 1) * split_size
        client_indices.append(indices[start:end])
    return client_indices


def split_selected_indices_iid(
    selected_indices: List[int],
    num_clients: int,
) -> List[List[int]]:
    """
    Split a pre-selected MNIST index pool into clients.
    This keeps indices in the original MNIST training set coordinate system,
    which is required by membership inference attack evaluation.
    """
    split_size = len(selected_indices) // num_clients
    client_indices: List[List[int]] = []
    for i in range(num_clients):
        start = i * split_size
        end = len(selected_indices) if i == num_clients - 1 else (i + 1) * split_size
        client_indices.append(selected_indices[start:end])
    return client_indices


def select_mnist_train_and_holdout_indices(
    dataset_size: int,
    train_size: int | None,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """
    Select a smaller MNIST training subset for FL training and keep the
    remaining MNIST training samples as same-distribution non-member samples.

    If train_size is None or <=0 or >= dataset_size, all samples are used for
    training and the holdout list is empty.
    """
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(dataset_size, generator=generator).tolist()

    if train_size is None or int(train_size) <= 0 or int(train_size) >= dataset_size:
        return perm, []

    train_size = int(train_size)
    train_pool = perm[:train_size]
    holdout_pool = perm[train_size:]
    return train_pool, holdout_pool


def tensor_to_json(name: str, tensor: torch.Tensor) -> dict:
    arr = tensor.detach().cpu().numpy()
    payload = base64.b64encode(arr.tobytes()).decode("ascii")
    return {"name": name, "shape": list(arr.shape), "dtype": str(arr.dtype), "encoding": "base64", "data": payload}


def tensor_from_json(obj: dict) -> torch.Tensor:
    dtype = np.dtype(obj["dtype"])
    shape = tuple(obj["shape"])
    raw = base64.b64decode(obj["data"].encode("ascii"))
    arr = np.frombuffer(raw, dtype=dtype).reshape(shape)
    return torch.from_numpy(arr.copy())


def state_dict_to_json(state_dict: Dict[str, torch.Tensor]) -> dict:
    return {"type": "state_dict", "tensors": {name: tensor_to_json(name, tensor) for name, tensor in state_dict.items()}}


def state_dict_from_json(obj: dict) -> Dict[str, torch.Tensor]:
    return {name: tensor_from_json(tensor_json) for name, tensor_json in obj["tensors"].items()}


def aggregate_plain_state_dicts_equal_weight(
    plain_updates: List[Dict[str, torch.Tensor]],
) -> Tuple[Dict[str, torch.Tensor], int]:
    """
    明文等权聚合路径。

    注意：这里故意保持与论文副本中当前研究成果一致，聚合权重固定为 1，
    即每个客户端贡献相同权重，而不是使用原始样本量加权。
    该函数仅用于 --he-mode none 的无保护 FedAvg baseline。
    """
    if not plain_updates:
        raise ValueError("No plain updates provided")

    total_weight = len(plain_updates)
    first_state = plain_updates[0]
    aggregated: Dict[str, torch.Tensor] = {}

    for name, tensor in first_state.items():
        acc = torch.zeros_like(tensor, dtype=torch.float32)
        for state in plain_updates:
            acc += state[name].detach().cpu().float()
        aggregated[name] = (acc / float(total_weight)).to(dtype=tensor.dtype)

    return aggregated, total_weight


def send_json(sock: socket.socket, payload: dict) -> int:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    packet = struct.pack("!I", len(data)) + data
    sock.sendall(packet)
    return len(packet)


def recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Socket connection closed unexpectedly")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_json(sock: socket.socket) -> tuple[dict, int]:
    header = recv_exact(sock, 4)
    (length,) = struct.unpack("!I", header)
    data = recv_exact(sock, length)
    return json.loads(data.decode("utf-8")), 4 + length


def get_model_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_model_state(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state, strict=True)


@torch.no_grad()
def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += float(loss.item()) * x.size(0)
        preds = logits.argmax(dim=1)
        total_correct += int((preds == y).sum().item())
        total_samples += x.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    acc = total_correct / max(total_samples, 1)
    return avg_loss, acc


def save_metrics_csv(metrics: List[dict], save_path: str) -> None:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not metrics:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)


def maybe_plot_metrics(metrics_csv_path: str, save_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[Plot] matplotlib not available, skipping plot generation.")
        return

    rows = []
    with open(metrics_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "round": int(row["round"]),
                "avg_client_loss": float(row["avg_client_loss"]),
                "avg_epsilon_estimate": float(row["avg_epsilon_estimate"]),
                "test_loss": float(row["test_loss"]),
                "test_acc": float(row["test_acc"]),
                "round_time_sec": float(row["round_time_sec"]),
                "upload_bytes": float(row["upload_bytes"]),
                "download_bytes": float(row["download_bytes"]),
            })

    if not rows:
        return

    save_root = Path(save_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    rounds = [r["round"] for r in rows]
    train_loss = [r["avg_client_loss"] for r in rows]
    test_loss = [r["test_loss"] for r in rows]
    test_acc = [r["test_acc"] for r in rows]
    eps = [r["avg_epsilon_estimate"] for r in rows]
    upload = [r["upload_bytes"] / 1024.0 for r in rows]
    download = [r["download_bytes"] / 1024.0 for r in rows]
    round_time = [r["round_time_sec"] for r in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, test_acc, marker="o")
    plt.xlabel("Round")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy vs Round")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_root / "test_accuracy.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, train_loss, marker="o", label="Train Loss")
    plt.plot(rounds, test_loss, marker="s", label="Test Loss")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.title("Train/Test Loss vs Round")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_root / "loss_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, eps, marker="o")
    plt.xlabel("Round")
    plt.ylabel("Average Epsilon Estimate")
    plt.title("Privacy Budget vs Round")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_root / "epsilon_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, upload, marker="o", label="Upload KB")
    plt.plot(rounds, download, marker="s", label="Download KB")
    plt.xlabel("Round")
    plt.ylabel("Communication (KB)")
    plt.title("Communication Cost vs Round")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_root / "communication_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, round_time, marker="o")
    plt.xlabel("Round")
    plt.ylabel("Time (s)")
    plt.title("Round Time vs Round")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_root / "round_time_curve.png", dpi=300)
    plt.close()


class FederatedClient:
    def __init__(
        self,
        client_id: int,
        cfg: FLConfig,
        he_context_blob: bytes,
        client_indices: List[int],
        upload_barrier=None,
        download_barrier=None,
    ):
        self.client_id = client_id
        self.cfg = cfg
        self.he_context_blob = he_context_blob
        self.upload_barrier = upload_barrier
        self.download_barrier = download_barrier
        self.device = torch.device("cpu")
        self.model = SimpleMLP(cfg.input_dim, cfg.num_classes).to(self.device)

        train_dataset = build_train_dataset(cfg, client_id)
        dataset = Subset(train_dataset, client_indices) if cfg.dataset.lower() == "mnist" else train_dataset
        self.loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

        self.ldp_cfg = LDPConfig(
            clip_norm=cfg.ldp_clip_norm,
            epsilon=cfg.ldp_epsilon,
            delta=cfg.ldp_delta,
            noise_multiplier=cfg.ldp_noise_multiplier,
        )

    def local_train(self) -> Tuple[Dict[str, torch.Tensor], int, float, dict]:
        sample_rate = self.cfg.batch_size / max(len(self.loader.dataset), 1)
        sample_rate = min(sample_rate, 1.0)

        if self.cfg.disable_ldp:
            state, avg_loss, steps = train_one_client_plain(
                model=self.model,
                dataloader=self.loader,
                local_epochs=self.cfg.local_epochs,
                lr=self.cfg.lr,
                device=self.device,
            )
            report = build_no_privacy_report(steps, sample_rate)
        else:
            state, avg_loss, steps = train_one_client_with_ldp(
                model=self.model,
                dataloader=self.loader,
                local_epochs=self.cfg.local_epochs,
                lr=self.cfg.lr,
                ldp_cfg=self.ldp_cfg,
                device=self.device,
            )
            report = build_privacy_report(steps, sample_rate, self.ldp_cfg)

        return state, len(self.loader.dataset), avg_loss, report

    def phase_upload(self, round_idx: int) -> int:
        total_bytes = 0
        with socket.create_connection((self.cfg.host, self.cfg.port), timeout=self.cfg.socket_timeout) as sock:
            sock.settimeout(self.cfg.socket_timeout)
            total_bytes += send_json(sock, {"type": "upload_request", "client_id": self.client_id, "round": round_idx})
            msg, recv_bytes = recv_json(sock)
            total_bytes += recv_bytes
            if msg.get("type") != "global_model":
                raise RuntimeError(f"Client {self.client_id}: unexpected message {msg}")

            global_state = state_dict_from_json(msg["payload"])
            load_model_state(self.model, global_state)

            local_state, num_samples, avg_loss, privacy_report = self.local_train()

            if self.cfg.he_mode == "none":
                payload = state_dict_to_json(local_state)
                payload_kind = "plain_state_dict"
            else:
                payload = encrypt_state_dict(
                    local_state,
                    context_blob=self.he_context_blob,
                    he_cfg=HEConfig(
                        scale=self.cfg.he_scale,
                        poly_modulus_degree=self.cfg.he_poly_modulus_degree,
                        plain_modulus=self.cfg.he_plain_modulus,
                    ),
                )
                payload_kind = "encrypted_state_dict"

            total_bytes += send_json(
                sock,
                {
                    "type": "client_update",
                    "client_id": self.client_id,
                    "round": round_idx,
                    "num_samples": num_samples,
                    "train_loss": avg_loss,
                    "privacy_report": privacy_report,
                    "payload_kind": payload_kind,
                    "payload": payload,
                },
            )
            ack, recv_bytes = recv_json(sock)
            total_bytes += recv_bytes
            if ack.get("type") != "ack_upload":
                raise RuntimeError(f"Client {self.client_id}: unexpected ack {ack}")
        return total_bytes

    def phase_download(self, round_idx: int) -> int:
        total_bytes = 0
        with socket.create_connection((self.cfg.host, self.cfg.port), timeout=self.cfg.socket_timeout) as sock:
            sock.settimeout(self.cfg.socket_timeout)
            total_bytes += send_json(sock, {"type": "download_request", "client_id": self.client_id, "round": round_idx})
            msg, recv_bytes = recv_json(sock)
            total_bytes += recv_bytes

            if self.cfg.he_mode == "none":
                if msg.get("type") != "aggregated_plain":
                    raise RuntimeError(f"Client {self.client_id}: unexpected aggregate message {msg}")
                aggregated_state = state_dict_from_json(msg["payload"])
            else:
                if msg.get("type") != "aggregated_cipher":
                    raise RuntimeError(f"Client {self.client_id}: unexpected aggregate message {msg}")
                aggregated_state = decrypt_aggregated_payload(
                    encrypted_payload=msg["payload"],
                    context_blob=self.he_context_blob,
                    total_weight=float(msg["total_weight"]),
                )

            load_model_state(self.model, aggregated_state)
            total_bytes += send_json(sock, {"type": "ack_download", "client_id": self.client_id, "round": round_idx, "status": "ok"})
        return total_bytes

    def run(self) -> None:
        for round_idx in range(self.cfg.rounds):
            self.phase_upload(round_idx)

            # Barrier 1: wait until all clients have completed upload.
            # Without this barrier, a fast client may send download_request while
            # the server is still waiting for other clients' upload_request, causing
            # "Server: unexpected request {'type': 'download_request', ...}".
            if self.upload_barrier is not None:
                self.upload_barrier.wait()

            self.phase_download(round_idx)

            # Barrier 2: wait until all clients have completed download before any
            # client starts the next round's upload_request. This keeps the server's
            # two-phase protocol synchronized across rounds.
            if self.download_barrier is not None:
                self.download_barrier.wait()

        print(f"[Client {self.client_id}] finished all rounds.")


class FederatedServer:
    def __init__(self, cfg: FLConfig, he_context_blob: bytes, ready_event: Event | None = None):
        self.cfg = cfg
        self.he_context_blob = he_context_blob
        self.ready_event = ready_event
        self.device = torch.device("cpu")
        self.model = SimpleMLP(cfg.input_dim, cfg.num_classes).to(self.device)
        self.global_state = get_model_state(self.model)
        self.pending_aggregate = None
        self.pending_total_weight = 0
        self.metrics: List[dict] = []
        self.test_loader = DataLoader(build_test_dataset(cfg), batch_size=256, shuffle=False)

    def serve(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
            server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_sock.bind((self.cfg.host, self.cfg.port))
            server_sock.listen(self.cfg.num_clients)
            server_sock.settimeout(self.cfg.socket_timeout)
            if self.ready_event is not None:
                self.ready_event.set()
            print(f"[Server] listening on {self.cfg.host}:{self.cfg.port}")
            print(f"[Server] he_mode={self.cfg.he_mode}, disable_ldp={self.cfg.disable_ldp}")
            print("[Server] aggregation weight policy: fixed equal weight = 1 for each client")

            for round_idx in range(self.cfg.rounds):
                round_start = time.time()
                encrypted_updates: List[Tuple[dict, int]] = []
                plain_updates: List[Dict[str, torch.Tensor]] = []
                round_losses: List[float] = []
                round_epsilons: List[float] = []
                upload_bytes = 0
                download_bytes = 0

                # Phase 1: send global model and receive client updates.
                for _ in range(self.cfg.num_clients):
                    conn, _ = server_sock.accept()
                    with conn:
                        conn.settimeout(self.cfg.socket_timeout)
                        req, recv_bytes = recv_json(conn)
                        upload_bytes += recv_bytes
                        if req.get("type") != "upload_request":
                            raise RuntimeError(f"Server: unexpected request {req}")

                        upload_bytes += send_json(
                            conn,
                            {
                                "type": "global_model",
                                "round": round_idx,
                                "payload": state_dict_to_json(self.global_state),
                            },
                        )

                        update_msg, recv_bytes = recv_json(conn)
                        upload_bytes += recv_bytes
                        if update_msg.get("type") != "client_update":
                            raise RuntimeError(f"Server: unexpected update {update_msg}")

                        # 重要：按照既有研究成果，聚合时每个客户端权重固定为 1，
                        # 不使用 update_msg["num_samples"] 作为聚合权重。
                        if self.cfg.he_mode == "none":
                            plain_updates.append(state_dict_from_json(update_msg["payload"]))
                        else:
                            encrypted_updates.append((update_msg["payload"], 1))

                        round_losses.append(float(update_msg["train_loss"]))
                        privacy_report = update_msg.get("privacy_report", {})
                        eps_value = privacy_report.get("epsilon_estimate", 0.0)
                        try:
                            eps_value = float(eps_value)
                        except Exception:
                            eps_value = float("inf")
                        round_epsilons.append(eps_value)

                        upload_bytes += send_json(conn, {"type": "ack_upload", "round": round_idx, "status": "ok"})

                # Aggregate updates.
                if self.cfg.he_mode == "none":
                    self.pending_aggregate, self.pending_total_weight = aggregate_plain_state_dicts_equal_weight(plain_updates)
                else:
                    self.pending_aggregate, self.pending_total_weight = aggregate_encrypted_payloads(
                        encrypted_updates,
                        context_blob=self.he_context_blob,
                    )

                # Phase 2: return aggregated model to clients.
                for _ in range(self.cfg.num_clients):
                    conn, _ = server_sock.accept()
                    with conn:
                        conn.settimeout(self.cfg.socket_timeout)
                        req, recv_bytes = recv_json(conn)
                        download_bytes += recv_bytes
                        if req.get("type") != "download_request":
                            raise RuntimeError(f"Server: unexpected request {req}")

                        if self.cfg.he_mode == "none":
                            response = {
                                "type": "aggregated_plain",
                                "round": round_idx,
                                "total_weight": self.pending_total_weight,
                                "payload": state_dict_to_json(self.pending_aggregate),
                            }
                        else:
                            response = {
                                "type": "aggregated_cipher",
                                "round": round_idx,
                                "total_weight": self.pending_total_weight,
                                "payload": self.pending_aggregate,
                            }

                        download_bytes += send_json(conn, response)
                        ack, recv_bytes = recv_json(conn)
                        download_bytes += recv_bytes
                        if ack.get("type") != "ack_download":
                            raise RuntimeError(f"Server: unexpected ack {ack}")

                # The simulation server keeps a synchronized global model for the next round.
                if self.cfg.he_mode == "none":
                    self.global_state = self.pending_aggregate
                else:
                    self.global_state = decrypt_aggregated_payload(
                        encrypted_payload=self.pending_aggregate,
                        context_blob=self.he_context_blob,
                        total_weight=float(self.pending_total_weight),
                    )
                load_model_state(self.model, self.global_state)

                mean_loss = sum(round_losses) / max(len(round_losses), 1)
                finite_eps = [x for x in round_epsilons if np.isfinite(x)]
                mean_epsilon = sum(finite_eps) / len(finite_eps) if finite_eps else float("inf")
                test_loss, test_acc = evaluate_model(self.model, self.test_loader, self.device)
                round_time = time.time() - round_start

                self.metrics.append(
                    {
                        "round": round_idx + 1,
                        "avg_client_loss": mean_loss,
                        "avg_epsilon_estimate": mean_epsilon,
                        "test_loss": test_loss,
                        "test_acc": test_acc,
                        "total_weight": self.pending_total_weight,
                        "round_time_sec": round_time,
                        "upload_bytes": upload_bytes,
                        "download_bytes": download_bytes,
                        "he_mode": self.cfg.he_mode,
                        "disable_ldp": self.cfg.disable_ldp,
                        "aggregation_weight_policy": "fixed_equal_weight_1",
                    }
                )

                num_received = len(plain_updates) if self.cfg.he_mode == "none" else len(encrypted_updates)
                print(
                    f"[Server] round {round_idx + 1}/{self.cfg.rounds} | "
                    f"clients={num_received} | "
                    f"avg_client_loss={mean_loss:.4f} | "
                    f"test_loss={test_loss:.4f} | "
                    f"test_acc={test_acc:.4f} | "
                    f"avg_epsilon={mean_epsilon:.4f} | "
                    f"time={round_time:.2f}s"
                )

            results_dir = Path(self.cfg.results_dir) / self.cfg.experiment_name
            results_dir.mkdir(parents=True, exist_ok=True)
            save_metrics_csv(self.metrics, str(results_dir / "metrics.csv"))
            with open(results_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(asdict(self.cfg), f, ensure_ascii=False, indent=2)
            torch.save(self.global_state, results_dir / "final_global_model.pt")
            print(f"[Server] Global model saved to {results_dir / 'final_global_model.pt'}")
            if self.cfg.save_plots:
                maybe_plot_metrics(str(results_dir / "metrics.csv"), str(results_dir))

            print(f"[Server] training completed. Results saved to: {results_dir}")


class SerializationTests(unittest.TestCase):
    def test_tensor_roundtrip(self):
        t = torch.randn(3, 4)
        obj = tensor_to_json("w", t)
        t2 = tensor_from_json(obj)
        self.assertTrue(torch.allclose(t, t2, atol=1e-7))


def run_simulation(cfg: FLConfig) -> None:
    prepare_data(cfg)

    # Reproducibility for model initialization and data splitting.
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    results_dir = Path(cfg.results_dir) / cfg.experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)

    if cfg.dataset.lower() == "mnist":
        train_dataset = build_train_dataset(cfg, client_id=0)
        train_pool_indices, train_holdout_indices = select_mnist_train_and_holdout_indices(
            dataset_size=len(train_dataset),
            train_size=cfg.mnist_train_size,
            seed=cfg.seed,
        )
        client_indices = split_selected_indices_iid(train_pool_indices, cfg.num_clients)
    else:
        client_indices = [[] for _ in range(cfg.num_clients)]
        train_pool_indices = []
        train_holdout_indices = []

    # Save membership indices for subsequent membership inference attack experiments.
    index_info = {
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "num_clients": cfg.num_clients,
        "aggregation_weight_policy": "fixed_equal_weight_1",
        "mnist_train_size": cfg.mnist_train_size,
        "train_pool_indices": sorted(train_pool_indices),
        "train_holdout_indices": sorted(train_holdout_indices),
        "clients": {str(i): client_indices[i] for i in range(cfg.num_clients)},
        "all_member_indices": sorted([idx for one_client in client_indices for idx in one_client]),
    }
    with open(results_dir / "client_indices.json", "w", encoding="utf-8") as f:
        json.dump(index_info, f, ensure_ascii=False, indent=2)
    print(f"[Data] client indices saved to {results_dir / 'client_indices.json'}")

    if cfg.he_mode == "none":
        he_blob = b"PLAINTEXT_MODE"
    elif cfg.he_mode == "mock":
        he_blob = b"MOCK_HE_CONTEXT"
    elif cfg.he_mode == "real":
        he_blob = build_he_context_blob(
            HEConfig(
                scale=cfg.he_scale,
                poly_modulus_degree=cfg.he_poly_modulus_degree,
                plain_modulus=cfg.he_plain_modulus,
            )
        )
    else:
        raise ValueError(f"Unsupported he_mode: {cfg.he_mode}")

    ready_event = Event()
    server = FederatedServer(cfg, he_blob, ready_event)
    server_proc = Process(target=server.serve, daemon=True)
    server_proc.start()

    if not ready_event.wait(timeout=20):
        raise RuntimeError("Server failed to start in time")
    time.sleep(0.5)

    upload_barrier = Barrier(cfg.num_clients)
    download_barrier = Barrier(cfg.num_clients)

    clients = [
        Process(
            target=FederatedClient(
                i,
                cfg,
                he_blob,
                client_indices[i],
                upload_barrier,
                download_barrier,
            ).run,
            daemon=True,
        )
        for i in range(cfg.num_clients)
    ]
    for p in clients:
        p.start()
    for p in clients:
        p.join()

    server_proc.join()


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FedAvg + LDP + HE with real-data evaluation and visualization")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--input-dim", type=int, default=20)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--samples-per-client", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--socket-timeout", type=float, default=20.0)

    parser.add_argument("--dataset", choices=["mnist", "synthetic"], default="mnist")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument(
        "--mnist-train-size",
        type=int,
        default=None,
        help=(
            "Use only this many MNIST training samples for FL training. "
            "The remaining MNIST training samples are saved as train_holdout_indices "
            "for membership inference non-member evaluation."
        ),
    )

    parser.add_argument("--ldp-clip-norm", type=float, default=1.0)
    parser.add_argument("--ldp-epsilon", type=float, default=4.0)
    parser.add_argument("--ldp-delta", type=float, default=1e-5)
    parser.add_argument("--ldp-noise-multiplier", type=float, default=None)
    parser.add_argument(
        "--disable-ldp",
        action="store_true",
        help="Disable both gradient clipping and Gaussian noise for the unprotected FedAvg baseline.",
    )
    parser.add_argument(
        "--he-mode",
        choices=["real", "mock", "none"],
        default="real",
        help="real: TenSEAL/BFV; mock: MockBFV; none: plaintext equal-weight aggregation.",
    )

    parser.add_argument("--he-scale", type=float, default=1000.0)
    parser.add_argument("--he-poly-modulus-degree", type=int, default=8192)
    parser.add_argument("--he-plain-modulus", type=int, default=1032193)

    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--experiment-name", default="mnist_demo")
    parser.add_argument("--no-save-plots", action="store_true")
    parser.add_argument("--test", action="store_true")
    return parser.parse_args(argv)


def main(argv: List[str]) -> None:
    args = parse_args(argv)
    if args.test:
        unittest.main(argv=[sys.argv[0]], exit=False)
        return

    input_dim = args.input_dim
    num_classes = args.num_classes
    if args.dataset == "mnist":
        input_dim = 28 * 28
        num_classes = 10

    cfg = FLConfig(
        host=args.host,
        port=args.port,
        num_clients=args.num_clients,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        input_dim=input_dim,
        num_classes=num_classes,
        samples_per_client=args.samples_per_client,
        seed=args.seed,
        socket_timeout=args.socket_timeout,
        dataset=args.dataset,
        data_root=args.data_root,
        download_data=args.download_data,
        mnist_train_size=args.mnist_train_size,
        ldp_clip_norm=args.ldp_clip_norm,
        ldp_epsilon=args.ldp_epsilon,
        ldp_delta=args.ldp_delta,
        ldp_noise_multiplier=args.ldp_noise_multiplier,
        disable_ldp=args.disable_ldp,
        he_mode=args.he_mode,
        he_scale=args.he_scale,
        he_poly_modulus_degree=args.he_poly_modulus_degree,
        he_plain_modulus=args.he_plain_modulus,
        results_dir=args.results_dir,
        experiment_name=args.experiment_name,
        save_plots=not args.no_save_plots,
    )

    print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))
    run_simulation(cfg)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main(sys.argv[1:])
