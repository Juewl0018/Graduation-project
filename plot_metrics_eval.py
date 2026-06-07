
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
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
    return rows


def plot_metrics(metrics_csv: str, output_dir: str):
    rows = load_metrics(metrics_csv)
    if not rows:
        raise ValueError("No rows found in metrics CSV")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

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
    plt.savefig(output / "test_accuracy.png", dpi=300)
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
    plt.savefig(output / "loss_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, eps, marker="o")
    plt.xlabel("Round")
    plt.ylabel("Average Epsilon Estimate")
    plt.title("Privacy Budget vs Round")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output / "epsilon_curve.png", dpi=300)
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
    plt.savefig(output / "communication_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, round_time, marker="o")
    plt.xlabel("Round")
    plt.ylabel("Time (s)")
    plt.title("Round Time vs Round")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output / "round_time_curve.png", dpi=300)
    plt.close()

    print(f"Plots saved to: {output}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python plot_metrics_eval.py <metrics_csv> <output_dir>")
        sys.exit(1)
    plot_metrics(sys.argv[1], sys.argv[2])
