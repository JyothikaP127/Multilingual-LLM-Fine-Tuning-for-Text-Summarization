"""Environment verification: Python, torch, CUDA/BF16, RAM, disk, installed
package versions vs. the pins in requirements.txt.

Does not train anything, does not download the dataset. Writes
results/qa/environment_report.json and prints a human-readable summary.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import repo_path  # noqa: E402
from src.utils.hardware import detect_hardware  # noqa: E402
from src.utils.reporting import save_json_report  # noqa: E402


def _read_requirements_pins() -> dict[str, str]:
    req_path = repo_path("requirements.txt")
    pins: dict[str, str] = {}
    pattern = re.compile(r"^([A-Za-z0-9_.\-]+)==([A-Za-z0-9_.\-+]+)$")
    for line in req_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        match = pattern.match(line)
        if match:
            pins[match.group(1).lower().replace("_", "-")] = match.group(2)
    return pins


def _installed_versions(package_names: list[str]) -> dict[str, str | None]:
    from importlib import metadata

    versions: dict[str, str | None] = {}
    for name in package_names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _system_memory_disk() -> dict:
    import psutil

    vm = psutil.virtual_memory()
    du = psutil.disk_usage(str(repo_path(".")))
    return {
        "total_ram_gb": round(vm.total / (1024**3), 2),
        "available_ram_gb": round(vm.available / (1024**3), 2),
        "disk_free_gb": round(du.free / (1024**3), 2),
    }


def main() -> None:
    hw = detect_hardware()
    mem_disk = _system_memory_disk()

    pinned = _read_requirements_pins()
    installed = _installed_versions(list(pinned.keys()) + ["torch"])

    version_mismatches = {
        name: {"pinned": pinned_version, "installed": installed.get(name)}
        for name, pinned_version in pinned.items()
        if installed.get(name) != pinned_version
    }

    report = {
        "hardware": hw.to_dict(),
        "memory_disk": mem_disk,
        "installed_versions": installed,
        "pinned_versions": pinned,
        "version_mismatches": version_mismatches,
    }

    out_path = save_json_report(report, repo_path("results/qa/environment_report.json"))

    print("=== Environment verification ===")
    print(f"Platform          : {hw.platform}")
    print(f"Python            : {hw.python_version}")
    print(f"CPU cores         : {hw.cpu_count}")
    print(f"Torch             : {hw.torch_version}")
    print(f"CUDA available    : {hw.cuda_available}")
    if hw.cuda_available:
        print(f"GPU               : {hw.gpu_name} ({hw.gpu_total_memory_gb:.1f} GB)")
        print(f"CUDA BF16 support : {hw.cuda_bf16_supported}")
    else:
        print("GPU               : none (CPU-only)")
    print(f"Total RAM         : {mem_disk['total_ram_gb']} GB")
    print(f"Available RAM     : {mem_disk['available_ram_gb']} GB")
    print(f"Disk free         : {mem_disk['disk_free_gb']} GB")

    if version_mismatches:
        print("\nPackage version mismatches vs. requirements.txt:")
        for name, versions in version_mismatches.items():
            print(f"  {name}: pinned={versions['pinned']} installed={versions['installed']}")
    else:
        print("\nAll pinned packages match installed versions.")

    print(f"\nReport written to: {out_path}")


if __name__ == "__main__":
    main()
