"""Full dependency + hardware audit.

Reports, for every package in requirements.txt: installed version, pinned
version, whether they match, and whether pip considers the environment
internally consistent (`pip check`). Also captures the hardware facts
(Python, torch, GPU/CUDA/BF16/FP16, RAM) needed to make batch-size and
precision decisions.

Does not train anything, does not download the dataset, does not modify any
config other than what has already been decided (max_source_length=768,
changed separately in configs/data.yaml).
"""
from __future__ import annotations

import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import repo_path  # noqa: E402
from src.utils.hardware import detect_hardware  # noqa: E402
from src.utils.reporting import save_json_report  # noqa: E402

# Packages the user asked to have explicitly audited, in the requested order.
AUDIT_PACKAGES = [
    "torch",
    "transformers",
    "peft",
    "datasets",
    "accelerate",
    "evaluate",
    "sentencepiece",
    "protobuf",
    "rouge_score",
    "bert_score",
    "pyarrow",
    "numpy",
    "pandas",
    "pyyaml",
    "huggingface_hub",
]

# importlib.metadata distribution names sometimes differ from the pip/import name.
DIST_NAME_OVERRIDES = {
    "rouge_score": "rouge-score",
    "bert_score": "bert-score",
    "pyyaml": "PyYAML",
    "huggingface_hub": "huggingface-hub",
}


def _read_requirements_pins() -> dict[str, str | None]:
    """Returns {package: pinned_version_or_None}. None means present in
    requirements.txt without an exact '==' pin (range-pinned or unpinned).
    """
    req_path = repo_path("requirements.txt")
    pins: dict[str, str | None] = {}
    exact = re.compile(r"^([A-Za-z0-9_.\-]+)==([A-Za-z0-9_.\-+]+)$")
    loose = re.compile(r"^([A-Za-z0-9_.\-]+)\s*(>=|<=|~=|<|>).*$")
    for raw_line in req_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        m_exact = exact.match(line)
        m_loose = loose.match(line)
        if m_exact:
            pins[m_exact.group(1).lower().replace("_", "-")] = m_exact.group(2)
        elif m_loose:
            pins[m_loose.group(1).lower().replace("_", "-")] = None
    return pins


def _installed_version(pkg_import_name: str) -> str | None:
    dist_name = DIST_NAME_OVERRIDES.get(pkg_import_name, pkg_import_name)
    try:
        return metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        return None


def _run_pip_check() -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "clean": proc.returncode == 0,
    }


def _fp16_support_note(hw) -> str:
    if hw.cuda_available:
        return "CUDA present: fp16 is hardware-supported, but NOT recommended for mT5 (documented NaN-loss risk under fp16 mixed precision, since mT5 was pretrained in bf16)."
    return "CPU-only: fp16 has no meaningful hardware acceleration on this machine; not recommended regardless of the mT5 NaN-risk caveat."


def main() -> None:
    hw = detect_hardware()
    import torch
    import psutil

    pins = _read_requirements_pins()
    pip_check = _run_pip_check()

    rows = []
    for pkg in AUDIT_PACKAGES:
        pinned = pins.get(pkg.lower().replace("_", "-"))
        installed = _installed_version(pkg)
        rows.append(
            {
                "package": pkg,
                "installed_version": installed,
                "pinned_version": pinned,
                "matches_pin": (installed == pinned) if pinned else None,
                "currently_range_or_unpinned": pinned is None,
            }
        )

    vm = psutil.virtual_memory()

    report = {
        "python_version": hw.python_version,
        "platform": hw.platform,
        "torch_version": hw.torch_version,
        "torch_cuda_build_version": torch.version.cuda,
        "cuda_available": hw.cuda_available,
        "gpu_name": hw.gpu_name,
        "gpu_total_memory_gb": hw.gpu_total_memory_gb,
        "cuda_bf16_supported": hw.cuda_bf16_supported,
        "fp16_note": _fp16_support_note(hw),
        "total_ram_gb": round(vm.total / (1024**3), 2),
        "available_ram_gb": round(vm.available / (1024**3), 2),
        "pip_check": pip_check,
        "dependency_audit": rows,
    }

    out_path = save_json_report(report, repo_path("results/qa/dependency_audit_report.json"))

    print("=== Hardware ===")
    print(f"Python            : {hw.python_version}")
    print(f"Platform          : {hw.platform}")
    print(f"Torch             : {hw.torch_version} (CUDA build: {torch.version.cuda})")
    print(f"CUDA available    : {hw.cuda_available}")
    print(f"GPU               : {hw.gpu_name or 'none (CPU-only)'}")
    if hw.cuda_available:
        print(f"GPU VRAM          : {hw.gpu_total_memory_gb:.1f} GB")
        print(f"BF16 support      : {hw.cuda_bf16_supported}")
    print(f"FP16 note         : {report['fp16_note']}")
    print(f"Total RAM         : {report['total_ram_gb']} GB")
    print(f"Available RAM     : {report['available_ram_gb']} GB")

    print("\n=== pip check ===")
    print("CLEAN - no broken requirements" if pip_check["clean"] else pip_check["stdout"])

    print("\n=== Dependency audit ===")
    header = f"{'package':<16}{'installed':<14}{'pinned':<14}{'match':<8}{'pin_status'}"
    print(header)
    print("-" * len(header))
    for row in rows:
        match = "n/a" if row["matches_pin"] is None else str(row["matches_pin"])
        pin_status = "RANGE/UNPINNED" if row["currently_range_or_unpinned"] else "exact-pinned"
        print(
            f"{row['package']:<16}{str(row['installed_version']):<14}"
            f"{str(row['pinned_version']):<14}{match:<8}{pin_status}"
        )

    print(f"\nReport written to: {out_path}")


if __name__ == "__main__":
    main()
