
from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

try:
    import tenseal as ts  # type: ignore
except Exception:
    ts = None


@dataclass
class HEConfig:
    scale: float = 1000.0
    poly_modulus_degree: int = 8192
    plain_modulus: int = 1032193


class MockBFVVector:
    def __init__(self, values: Sequence[int]):
        self.values = [int(v) for v in values]

    def __add__(self, other: "MockBFVVector") -> "MockBFVVector":
        if len(self.values) != len(other.values):
            raise ValueError("Ciphertext size mismatch")
        return MockBFVVector([a + b for a, b in zip(self.values, other.values)])

    def __mul__(self, scalar: int) -> "MockBFVVector":
        return MockBFVVector([int(v) * int(scalar) for v in self.values])

    __rmul__ = __mul__

    def serialize(self) -> bytes:
        return json.dumps({"values": self.values}).encode("utf-8")

    @classmethod
    def deserialize(cls, raw: bytes) -> "MockBFVVector":
        obj = json.loads(raw.decode("utf-8"))
        return cls(obj["values"])

    def decrypt(self) -> List[int]:
        return list(self.values)


def build_he_context_blob(he_cfg: HEConfig) -> bytes:
    if ts is None:
        return b"MOCK_HE_CONTEXT"

    try:
        context = ts.context(
            ts.SCHEME_TYPE.BFV,
            poly_modulus_degree=he_cfg.poly_modulus_degree,
            plain_modulus=he_cfg.plain_modulus,
        )
    except TypeError:
        context = ts.context(ts.SCHEME_TYPE.BFV, he_cfg.poly_modulus_degree, he_cfg.plain_modulus)

    try:
        context.generate_galois_keys()
    except Exception:
        pass

    try:
        return context.serialize(save_secret_key=True)
    except TypeError:
        return context.serialize()


def _load_context(context_blob: bytes):
    if ts is None or context_blob == b"MOCK_HE_CONTEXT":
        return None
    try:
        return ts.context_from(context_blob)
    except Exception:
        return None


def _quantize_tensor(tensor: torch.Tensor, scale: float) -> List[int]:
    arr = tensor.detach().cpu().numpy().astype(np.float64)
    return np.rint(arr * scale).astype(np.int64).ravel().tolist()


def _dequantize_tensor(
    values: Sequence[int],
    shape: Tuple[int, ...],
    scale: float,
    total_weight: float,
) -> torch.Tensor:
    arr = np.asarray(list(values), dtype=np.float64).reshape(shape)
    arr = arr / float(scale)
    arr = arr / float(total_weight)
    return torch.from_numpy(arr.astype(np.float32))


def _encrypt_int_vector(values: Sequence[int], context_blob: bytes):
    context = _load_context(context_blob)
    if context is None:
        return MockBFVVector(values)
    try:
        return ts.bfv_vector(context, list(map(int, values)))
    except Exception:
        return MockBFVVector(values)


def _serialize_cipher(cipher_obj) -> str:
    return base64.b64encode(cipher_obj.serialize()).decode("ascii")


def _deserialize_cipher(ciphertext_b64: str, context_blob: bytes):
    raw = base64.b64decode(ciphertext_b64.encode("ascii"))
    context = _load_context(context_blob)

    if context is None:
        return MockBFVVector.deserialize(raw)

    try:
        if hasattr(ts, "bfv_vector_from"):
            return ts.bfv_vector_from(context, raw)
    except Exception:
        pass

    try:
        if hasattr(ts, "lazy_bfv_vector_from"):
            return ts.lazy_bfv_vector_from(context, raw)
    except Exception:
        pass

    return MockBFVVector.deserialize(raw)


def _decrypt_cipher(cipher_obj) -> List[int]:
    if isinstance(cipher_obj, MockBFVVector):
        return cipher_obj.decrypt()
    if hasattr(cipher_obj, "decrypt"):
        return [int(v) for v in cipher_obj.decrypt()]
    raise TypeError("Unsupported cipher object")


def encrypt_state_dict(
    state_dict: Dict[str, torch.Tensor],
    context_blob: bytes,
    he_cfg: HEConfig,
) -> dict:
    payload = {}
    for name, tensor in state_dict.items():
        values = _quantize_tensor(tensor, he_cfg.scale)
        cipher = _encrypt_int_vector(values, context_blob)
        payload[name] = {
            "shape": list(tensor.shape),
            "scale": float(he_cfg.scale),
            "ciphertext": _serialize_cipher(cipher),
        }
    return {"type": "encrypted_state_dict", "tensors": payload}


def aggregate_encrypted_payloads(
    encrypted_updates: List[Tuple[dict, int]],
    context_blob: bytes,
) -> Tuple[dict, int]:
    if not encrypted_updates:
        raise ValueError("No encrypted updates provided")

    total_weight = int(sum(weight for _, weight in encrypted_updates))
    if total_weight <= 0:
        raise ValueError("Total weight must be positive")

    first_payload = encrypted_updates[0][0]
    aggregated = {}

    for name, meta in first_payload["tensors"].items():
        acc = None
        for payload, weight in encrypted_updates:
            cur = payload["tensors"][name]
            cipher = _deserialize_cipher(cur["ciphertext"], context_blob)
            weighted = cipher * int(weight)
            acc = weighted if acc is None else acc + weighted

        aggregated[name] = {
            "shape": list(meta["shape"]),
            "scale": float(meta["scale"]),
            "ciphertext": _serialize_cipher(acc),
        }

    return {"type": "encrypted_state_dict", "tensors": aggregated}, total_weight


def decrypt_aggregated_payload(
    encrypted_payload: dict,
    context_blob: bytes,
    total_weight: float,
) -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {}

    for name, item in encrypted_payload["tensors"].items():
        cipher = _deserialize_cipher(item["ciphertext"], context_blob)
        values = _decrypt_cipher(cipher)
        state[name] = _dequantize_tensor(
            values=values,
            shape=tuple(item["shape"]),
            scale=float(item["scale"]),
            total_weight=total_weight,
        )

    return state



def fedavg_plain_aggregate(
    state_and_weights: List[Tuple[Dict[str, torch.Tensor], int]],
) -> Dict[str, torch.Tensor]:
    if not state_and_weights:
        raise ValueError("No updates provided")

    total_weight = float(sum(weight for _, weight in state_and_weights))
    if total_weight <= 0:
        raise ValueError("Total weight must be positive")

    first_state = state_and_weights[0][0]
    avg_state: Dict[str, torch.Tensor] = {
        k: torch.zeros_like(v) for k, v in first_state.items()
    }

    for state, weight in state_and_weights:
        w = float(weight)
        for k in avg_state.keys():
            avg_state[k] += state[k] * w

    for k in avg_state.keys():
        avg_state[k] /= total_weight

    return avg_state
