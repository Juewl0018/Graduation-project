from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from torchvision import datasets, transforms
except Exception as exc:
    raise ImportError("This script requires torchvision. Please install torchvision first.") from exc


class SimpleMLP(nn.Module):
    """Keep the same model structure as fedavg_design_mnist_eval_exp.py."""

    def __init__(self, input_dim: int = 784, num_classes: int = 10):
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


def flatten_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.view(-1)


def mnist_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(flatten_tensor),
    ])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_member_indices(exp_dir: Path) -> List[int]:
    path = exp_dir / "client_indices.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing client_indices.json: {path}")
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if "all_member_indices" in obj:
        return [int(x) for x in obj["all_member_indices"]]
    if "clients" in obj:
        merged: List[int] = []
        for v in obj["clients"].values():
            merged.extend(int(x) for x in v)
        return sorted(set(merged))
    raise KeyError(f"client_indices.json has no all_member_indices or clients field: {path}")


def load_train_holdout_indices(exp_dir: Path) -> List[int]:
    """
    Load same-distribution non-member samples from MNIST train holdout.
    This requires training with fedavg_design_mnist_eval_exp_mia.py and
    --mnist-train-size.
    """
    path = exp_dir / "client_indices.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing client_indices.json: {path}")
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    values = obj.get("train_holdout_indices", [])
    return [int(x) for x in values]


def sample_indices(indices: Sequence[int], max_samples: int | None, seed: int) -> List[int]:
    indices = list(indices)
    rng = random.Random(seed)
    rng.shuffle(indices)
    if max_samples is not None and max_samples > 0:
        indices = indices[: min(max_samples, len(indices))]
    return indices


def load_model(exp_dir: Path, input_dim: int, num_classes: int, device: torch.device) -> nn.Module:
    model_path = exp_dir / "final_global_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing final_global_model.pt: {model_path}")

    model = SimpleMLP(input_dim=input_dim, num_classes=num_classes).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def extract_attack_features(
    model: nn.Module,
    dataset: Dataset,
    indices: Sequence[int],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """
    Extract black-box membership inference features from model outputs.

    Features:
    - sorted class probabilities, descending order, length = num_classes
    - true_label_prob
    - max_prob
    - entropy
    - cross_entropy_loss
    - correct prediction indicator
    """
    subset = Subset(dataset, list(indices))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss(reduction="none")

    all_features: List[np.ndarray] = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        losses = criterion(logits, y)

        sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
        true_probs = probs.gather(1, y.view(-1, 1)).squeeze(1)
        max_probs = probs.max(dim=1).values
        entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1)
        correct = (probs.argmax(dim=1) == y).float()

        feat = torch.cat(
            [
                sorted_probs,
                true_probs.view(-1, 1),
                max_probs.view(-1, 1),
                entropy.view(-1, 1),
                losses.view(-1, 1),
                correct.view(-1, 1),
            ],
            dim=1,
        )
        all_features.append(feat.detach().cpu().numpy())

    if not all_features:
        raise ValueError("No features extracted. Please check indices and datasets.")
    return np.vstack(all_features).astype(np.float32)


class AttackMLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def standardize_train_test(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (x_train - mean) / std, (x_test - mean) / std, mean.squeeze(0), std.squeeze(0)


def train_attack_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
) -> AttackMLP:
    model = AttackMLP(in_dim=x_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    x_tensor = torch.from_numpy(x_train.astype(np.float32))
    y_tensor = torch.from_numpy(y_train.astype(np.int64))
    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total = 0
        correct = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * xb.size(0)
            total += xb.size(0)
            correct += int((logits.argmax(dim=1) == yb).sum().item())

        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(
                f"[Attack Train] epoch {epoch + 1}/{epochs} | "
                f"loss={total_loss / max(total, 1):.4f} | "
                f"acc={correct / max(total, 1):.4f}"
            )

    return model


@torch.no_grad()
def predict_attack_scores(
    model: AttackMLP,
    x: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    x_tensor = torch.from_numpy(x.astype(np.float32))
    loader = DataLoader(x_tensor, batch_size=batch_size, shuffle=False)
    scores: List[np.ndarray] = []
    preds: List[np.ndarray] = []

    for xb in loader:
        xb = xb.to(device)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)
        scores.append(probs[:, 1].detach().cpu().numpy())
        preds.append(logits.argmax(dim=1).detach().cpu().numpy())

    return np.concatenate(scores), np.concatenate(preds)


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    scores = scores.astype(np.float64)
    pos = int(labels.sum())
    neg = int(len(labels) - pos)
    if pos == 0 or neg == 0:
        return float("nan")

    order = np.argsort(-scores)
    sorted_labels = labels[order]
    tps = np.cumsum(sorted_labels == 1)
    fps = np.cumsum(sorted_labels == 0)
    tpr = tps / max(pos, 1)
    fpr = fps / max(neg, 1)

    tpr = np.concatenate([[0.0], tpr, [1.0]])
    fpr = np.concatenate([[0.0], fpr, [1.0]])
    return float(np.trapz(tpr, fpr))


def roc_points(labels: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    labels = labels.astype(np.int64)
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    pos = int(labels.sum())
    neg = int(len(labels) - pos)
    tps = np.cumsum(sorted_labels == 1)
    fps = np.cumsum(sorted_labels == 0)
    tpr = tps / max(pos, 1)
    fpr = fps / max(neg, 1)
    return np.concatenate([[0.0], fpr, [1.0]]), np.concatenate([[0.0], tpr, [1.0]])


def save_roc_plot(labels: np.ndarray, scores: np.ndarray, auc: float, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[Plot] matplotlib not available, skip ROC plot.")
        return

    fpr, tpr = roc_points(labels, scores)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"Attack ROC, AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Membership Inference Attack ROC")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_metrics_csv(metrics: dict, out_path: Path) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Black-box membership inference attack using a shadow model.")
    parser.add_argument("--target-dir", type=str, required=True, help="Experiment dir of target model, e.g. results/target_fedavg_plain")
    parser.add_argument("--shadow-dir", type=str, required=True, help="Experiment dir of shadow model, e.g. results/shadow_fedavg_plain")
    parser.add_argument("--output-dir", type=str, required=True, help="Output dir for attack results")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--input-dim", type=int, default=784)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--max-member-samples", type=int, default=5000)
    parser.add_argument("--max-nonmember-samples", type=int, default=5000)
    parser.add_argument(
        "--nonmember-source",
        choices=["test", "train_holdout"],
        default="train_holdout",
        help=(
            "test: use MNIST official test set as non-members; "
            "train_holdout: use unused MNIST training samples saved by --mnist-train-size. "
            "train_holdout is preferred for MIA because it removes train/test distribution shift."
        ),
    )
    parser.add_argument("--feature-batch-size", type=int, default=512)
    parser.add_argument("--attack-batch-size", type=int, default=256)
    parser.add_argument("--attack-epochs", type=int, default=50)
    parser.add_argument("--attack-lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"[Info] device={device}")

    target_dir = Path(args.target_dir)
    shadow_dir = Path(args.shadow_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = datasets.MNIST(root=args.data_root, train=True, download=args.download_data, transform=mnist_transform())
    test_dataset = datasets.MNIST(root=args.data_root, train=False, download=args.download_data, transform=mnist_transform())

    target_model = load_model(target_dir, args.input_dim, args.num_classes, device)
    shadow_model = load_model(shadow_dir, args.input_dim, args.num_classes, device)

    shadow_member_indices = sample_indices(load_member_indices(shadow_dir), args.max_member_samples, args.seed + 1)
    target_member_indices = sample_indices(load_member_indices(target_dir), args.max_member_samples, args.seed + 2)

    if args.nonmember_source == "train_holdout":
        shadow_nonmember_pool = load_train_holdout_indices(shadow_dir)
        target_nonmember_pool = load_train_holdout_indices(target_dir)
        if len(shadow_nonmember_pool) == 0 or len(target_nonmember_pool) == 0:
            raise ValueError(
                "train_holdout non-member source was requested, but train_holdout_indices is empty. "
                "Please retrain target/shadow with fedavg_design_mnist_eval_exp_mia.py "
                "and a command-line option such as --mnist-train-size 2000, "
                "or use --nonmember-source test."
            )
        shadow_nonmember_dataset = train_dataset
        target_nonmember_dataset = train_dataset
    else:
        shadow_nonmember_pool = list(range(len(test_dataset)))
        target_nonmember_pool = list(range(len(test_dataset)))
        shadow_nonmember_dataset = test_dataset
        target_nonmember_dataset = test_dataset

    shadow_nonmember_indices = sample_indices(shadow_nonmember_pool, args.max_nonmember_samples, args.seed + 3)
    target_nonmember_indices = sample_indices(target_nonmember_pool, args.max_nonmember_samples, args.seed + 4)

    print(f"[Data] nonmember_source={args.nonmember_source}")
    print(f"[Data] shadow members={len(shadow_member_indices)}, shadow nonmembers={len(shadow_nonmember_indices)}")
    print(f"[Data] target members={len(target_member_indices)}, target nonmembers={len(target_nonmember_indices)}")

    print("[Feature] Extracting shadow member features...")
    x_shadow_member = extract_attack_features(shadow_model, train_dataset, shadow_member_indices, args.feature_batch_size, device)
    print("[Feature] Extracting shadow non-member features...")
    x_shadow_nonmember = extract_attack_features(shadow_model, shadow_nonmember_dataset, shadow_nonmember_indices, args.feature_batch_size, device)

    x_attack_train = np.vstack([x_shadow_member, x_shadow_nonmember]).astype(np.float32)
    y_attack_train = np.concatenate([
        np.ones(len(x_shadow_member), dtype=np.int64),
        np.zeros(len(x_shadow_nonmember), dtype=np.int64),
    ])

    print("[Feature] Extracting target member features...")
    x_target_member = extract_attack_features(target_model, train_dataset, target_member_indices, args.feature_batch_size, device)
    print("[Feature] Extracting target non-member features...")
    x_target_nonmember = extract_attack_features(target_model, target_nonmember_dataset, target_nonmember_indices, args.feature_batch_size, device)

    x_attack_test = np.vstack([x_target_member, x_target_nonmember]).astype(np.float32)
    y_attack_test = np.concatenate([
        np.ones(len(x_target_member), dtype=np.int64),
        np.zeros(len(x_target_nonmember), dtype=np.int64),
    ])

    x_attack_train, x_attack_test, feat_mean, feat_std = standardize_train_test(x_attack_train, x_attack_test)

    attack_model = train_attack_model(
        x_train=x_attack_train,
        y_train=y_attack_train,
        epochs=args.attack_epochs,
        lr=args.attack_lr,
        batch_size=args.attack_batch_size,
        device=device,
    )

    scores, preds = predict_attack_scores(attack_model, x_attack_test, args.attack_batch_size, device)
    acc = float((preds == y_attack_test).mean())
    auc = binary_auc(y_attack_test, scores)
    advantage = float(2.0 * auc - 1.0) if not math.isnan(auc) else float("nan")

    tp = int(((preds == 1) & (y_attack_test == 1)).sum())
    tn = int(((preds == 0) & (y_attack_test == 0)).sum())
    fp = int(((preds == 1) & (y_attack_test == 0)).sum())
    fn = int(((preds == 0) & (y_attack_test == 1)).sum())
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)

    metrics = {
        "target_dir": str(target_dir),
        "shadow_dir": str(shadow_dir),
        "num_shadow_member": len(x_shadow_member),
        "num_shadow_nonmember": len(x_shadow_nonmember),
        "num_target_member": len(x_target_member),
        "num_target_nonmember": len(x_target_nonmember),
        "nonmember_source": args.nonmember_source,
        "attack_accuracy": acc,
        "attack_auc": auc,
        "attack_advantage_2auc_minus_1": advantage,
        "tpr": tpr,
        "fpr": fpr,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }

    save_metrics_csv(metrics, output_dir / "attack_metrics.csv")
    save_roc_plot(y_attack_test, scores, auc, output_dir / "attack_roc.png")

    np.savez(
        output_dir / "attack_predictions.npz",
        labels=y_attack_test,
        scores=scores,
        preds=preds,
        feature_mean=feat_mean,
        feature_std=feat_std,
    )

    print("\n========== Attack Result ==========")
    print(f"Attack Accuracy : {acc:.4f}")
    print(f"Attack AUC      : {auc:.4f}")
    print(f"Attack Advantage: {advantage:.4f}")
    print(f"TPR             : {tpr:.4f}")
    print(f"FPR             : {fpr:.4f}")
    print(f"Saved to        : {output_dir}")


if __name__ == "__main__":
    main()
