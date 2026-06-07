
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


MAIN_FILE = "FedAvg design.py"
LDP_FILE = "ldp_module.py"
HE_FILE = "he_module.py"


def print_title(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def check_files(base_dir: Path) -> None:
    print_title("Step 1: Check required files")
    required = [MAIN_FILE, LDP_FILE, HE_FILE]
    missing = []

    for name in required:
        file_path = base_dir / name
        if file_path.exists():
            print(f"[OK] Found: {file_path}")
        else:
            print(f"[ERROR] Missing: {file_path}")
            missing.append(name)

    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")


def check_import(module_path: Path, module_name: str) -> None:
    print(f"[CHECK] Importing {module_name} from {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {module_name}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    print(f"[OK] Imported {module_name} successfully")


def run_command(cmd: list[str], cwd: Path) -> None:
    print(f"[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd), text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")
    print("[OK] Command finished successfully")


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    print_title("Hybrid FL prototype quick test")
    print(f"Working directory: {base_dir}")
    print(f"Python executable: {sys.executable}")

    check_files(base_dir)

    print_title("Step 2: Import check")
    check_import(base_dir / LDP_FILE, "ldp_module")
    check_import(base_dir / HE_FILE, "he_module")

    print_title("Step 3: Run built-in unit tests")
    run_command([sys.executable, MAIN_FILE, "--test"], cwd=base_dir)

    print_title("Step 4: Run smoke test")
    run_command(
        [
            sys.executable,
            MAIN_FILE,
            "--num-clients", "2",
            "--rounds", "1",
            "--local-epochs", "1",
            "--batch-size", "16",
        ],
        cwd=base_dir,
    )

    print_title("All checks passed")
    print("The prototype completed import check, unit tests, and a smoke test successfully.")


if __name__ == "__main__":
    main()
