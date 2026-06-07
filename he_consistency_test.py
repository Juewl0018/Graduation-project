
from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn


@dataclass
class TestConfig:
    module_name: str = "he_module_eval"
    mode: str = "auto"          # auto | mock
    input_dim: int = 28 * 28
    num_classes: int = 10
    scale: float = 1000.0
    poly_modulus_degree: int = 8192
    plain_modulus: int = 1032193
    seed: int = 42
    weight1: int = 300
    weight2: int = 500
    tolerance: float = 5e-3
    report_path: str = "./he_consistency_report.json"


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


def build_random_state(seed: int, input_dim: int, num_classes: int) -> Dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    model = SimpleMLP(input_dim=input_dim, num_classes=num_classes)
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def state_add_weighted(a: Dict[str, torch.Tensor], wa: int, b: Dict[str, torch.Tensor], wb: int) -> Dict[str, torch.Tensor]:
    total = wa + wb
    out = {}
    for k in a.keys():
        out[k] = (a[k] * float(wa) + b[k] * float(wb)) / float(total)
    return out


def compare_states(ref: Dict[str, torch.Tensor], got: Dict[str, torch.Tensor]) -> Tuple[dict, float, float]:
    per_layer = {}
    global_max = 0.0
    weighted_sum = 0.0
    count = 0

    for name in ref.keys():
        diff = (ref[name] - got[name]).abs()
        max_abs = float(diff.max().item())
        mean_abs = float(diff.mean().item())
        per_layer[name] = {
            "shape": list(ref[name].shape),
            "max_abs_diff": max_abs,
            "mean_abs_diff": mean_abs,
        }
        global_max = max(global_max, max_abs)
        weighted_sum += float(diff.sum().item())
        count += diff.numel()

    global_mean = weighted_sum / max(count, 1)
    return per_layer, global_max, global_mean


def print_summary(title: str, global_max: float, global_mean: float, top_items: list[tuple[str, dict]]) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(f"Global max abs diff : {global_max:.10f}")
    print(f"Global mean abs diff: {global_mean:.10f}")
    print("Top layers by max diff:")
    for name, info in top_items:
        print(f"  {name:<20} max={info['max_abs_diff']:.10f} mean={info['mean_abs_diff']:.10f} shape={info['shape']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="HE consistency test: plaintext aggregation vs encrypted aggregation")
    parser.add_argument("--module-name", default="he_module_eval")
    parser.add_argument("--mode", choices=["auto", "mock"], default="auto")
    parser.add_argument("--input-dim", type=int, default=28 * 28)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--scale", type=float, default=1000.0)
    parser.add_argument("--poly-modulus-degree", type=int, default=8192)
    parser.add_argument("--plain-modulus", type=int, default=1032193)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight1", type=int, default=300)
    parser.add_argument("--weight2", type=int, default=500)
    parser.add_argument("--tolerance", type=float, default=5e-3)
    parser.add_argument("--report-path", default="./he_consistency_report.json")
    args = parser.parse_args()

    cfg = TestConfig(**vars(args))

    he = importlib.import_module(cfg.module_name)

    # Force mock path if requested.
    if cfg.mode == "mock":
        he.ts = None
        context_blob = b"MOCK_HE_CONTEXT"
        print("[Mode] Using MOCK HE path")
    else:
        print("[Mode] Using AUTO path (real TenSEAL if available, otherwise fallback)")
        context_blob = he.build_he_context_blob(
            he.HEConfig(
                scale=cfg.scale,
                poly_modulus_degree=cfg.poly_modulus_degree,
                plain_modulus=cfg.plain_modulus,
            )
        )

    he_cfg = he.HEConfig(
        scale=cfg.scale,
        poly_modulus_degree=cfg.poly_modulus_degree,
        plain_modulus=cfg.plain_modulus,
    )

    # Build two random model states.
    state1 = build_random_state(seed=cfg.seed, input_dim=cfg.input_dim, num_classes=cfg.num_classes)
    state2 = build_random_state(seed=cfg.seed + 1, input_dim=cfg.input_dim, num_classes=cfg.num_classes)

    # ------------------------------------------------------------
    # Test 1: Single-state encrypt -> decrypt roundtrip
    # ------------------------------------------------------------
    enc1 = he.encrypt_state_dict(state1, context_blob=context_blob, he_cfg=he_cfg)
    first_name = next(iter(enc1["tensors"]))
    cipher_obj = he._deserialize_cipher(enc1["tensors"][first_name]["ciphertext"], context_blob)
    print("cipher object type:", type(cipher_obj))
    print("context is mock:", context_blob == b"MOCK_HE_CONTEXT")
    dec1 = he.decrypt_aggregated_payload(enc1, context_blob=context_blob, total_weight=1.0)

    per_layer_single, global_max_single, global_mean_single = compare_states(state1, dec1)
    top_single = sorted(per_layer_single.items(), key=lambda x: x[1]["max_abs_diff"], reverse=True)[:5]
    print_summary("Test 1: Single Model Roundtrip", global_max_single, global_mean_single, top_single)

    # ------------------------------------------------------------
    # Test 2: Two-client weighted aggregation consistency
    # ------------------------------------------------------------
    plain_avg = state_add_weighted(state1, cfg.weight1, state2, cfg.weight2)

    enc2 = he.encrypt_state_dict(state2, context_blob=context_blob, he_cfg=he_cfg)
    aggregated_enc, total_weight = he.aggregate_encrypted_payloads(
        [(enc1, cfg.weight1), (enc2, cfg.weight2)],
        context_blob=context_blob,
    )
    dec_avg = he.decrypt_aggregated_payload(
        aggregated_enc,
        context_blob=context_blob,
        total_weight=float(total_weight),
    )

    per_layer_agg, global_max_agg, global_mean_agg = compare_states(plain_avg, dec_avg)
    top_agg = sorted(per_layer_agg.items(), key=lambda x: x[1]["max_abs_diff"], reverse=True)[:5]
    print_summary("Test 2: Two-Client Weighted Aggregation", global_max_agg, global_mean_agg, top_agg)

    # ------------------------------------------------------------
    # Overall judgment
    # ------------------------------------------------------------
    single_pass = global_max_single <= cfg.tolerance
    agg_pass = global_max_agg <= cfg.tolerance

    print("\n" + "=" * 72)
    print("Overall Judgment")
    print("=" * 72)
    print(f"Tolerance            : {cfg.tolerance}")
    print(f"Single roundtrip pass: {single_pass}")
    print(f"Aggregation pass     : {agg_pass}")

    if single_pass and agg_pass:
        print("Conclusion: current HE path is numerically consistent within tolerance.")
    elif single_pass and not agg_pass:
        print("Conclusion: single encrypt/decrypt looks OK, but aggregation path is inconsistent.")
    elif (not single_pass) and agg_pass:
        print("Conclusion: rare case; check single-state encoding/decoding carefully.")
    else:
        print("Conclusion: current HE path is NOT numerically consistent. Focus on encoding, sign handling, or dequantization.")

    report = {
        "config": asdict(cfg),
        "single_test": {
            "global_max_abs_diff": global_max_single,
            "global_mean_abs_diff": global_mean_single,
            "pass": single_pass,
            "top_layers": dict(top_single),
        },
        "aggregation_test": {
            "global_max_abs_diff": global_max_agg,
            "global_mean_abs_diff": global_mean_agg,
            "pass": agg_pass,
            "top_layers": dict(top_agg),
        },
        "overall": {
            "single_pass": single_pass,
            "aggregation_pass": agg_pass,
        },
    }

    report_path = Path(cfg.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport saved to: {report_path.resolve()}")


if __name__ == "__main__":
    main()
