from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from common import MODELS, ffmpeg_path


def command_output(command: list[str], timeout: int = 15) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        return result.stdout.strip() if result.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def normalized_system() -> str:
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    if sys.platform.startswith("linux"):
        return "linux"
    return sys.platform


def normalized_architecture() -> str:
    value = platform.machine().lower()
    if value in {"amd64", "x86_64", "x64"}:
        return "x64"
    if value in {"arm64", "aarch64"}:
        return "arm64"
    return value or "unknown"


def nvidia_gpus() -> list[dict[str, Any]]:
    output = command_output([
        "nvidia-smi", "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ])
    devices = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            memory = int(parts[1])
        except ValueError:
            memory = None
        devices.append({
            "name": parts[0],
            "vendor": "nvidia",
            "vram_mib": memory,
            "driver": parts[2] if len(parts) > 2 else None,
        })
    return devices


def windows_details() -> tuple[int | None, str | None, list[dict[str, Any]]]:
    script = (
        "$os = Get-CimInstance Win32_OperatingSystem; "
        "$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1; "
        "$gpu = Get-CimInstance Win32_VideoController; "
        "@{ram_bytes=[int64]$os.TotalVisibleMemorySize*1024; cpu=$cpu.Name; "
        "gpu=@($gpu | ForEach-Object {@{name=$_.Name; vram_bytes=[int64]$_.AdapterRAM}})} | "
        "ConvertTo-Json -Depth 4 -Compress"
    )
    output = command_output(["powershell.exe", "-NoProfile", "-Command", script])
    if not output:
        return None, None, []
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return None, None, []
    devices = []
    for item in payload.get("gpu") or []:
        vram_bytes = item.get("vram_bytes") or 0
        devices.append({
            "name": item.get("name") or "Unknown GPU",
            "vendor": "unknown",
            "vram_mib": round(vram_bytes / 1024 / 1024) if vram_bytes > 0 else None,
            "vram_note": "Windows AdapterRAM can be inaccurate; nvidia-smi takes precedence when available.",
        })
    return payload.get("ram_bytes"), payload.get("cpu"), devices


def macos_details() -> tuple[int | None, str | None, list[dict[str, Any]]]:
    memory = command_output(["sysctl", "-n", "hw.memsize"])
    cpu = command_output(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor()
    devices = []
    output = command_output(["system_profiler", "SPDisplaysDataType", "-json"], timeout=30)
    if output:
        try:
            payload = json.loads(output)
            for item in payload.get("SPDisplaysDataType", []):
                devices.append({
                    "name": item.get("sppci_model") or item.get("_name") or "Apple GPU",
                    "vendor": "apple",
                    "vram_mib": None,
                    "memory_note": "Apple Silicon uses unified memory.",
                })
        except json.JSONDecodeError:
            pass
    return int(memory) if memory.isdigit() else None, cpu or None, devices


def linux_details() -> tuple[int | None, str | None, list[dict[str, Any]]]:
    memory = None
    cpu = platform.processor() or None
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
        for line in meminfo.splitlines():
            if line.startswith("MemTotal:"):
                memory = int(line.split()[1]) * 1024
                break
    except (OSError, ValueError, IndexError):
        pass
    try:
        cpu_info = Path("/proc/cpuinfo").read_text(encoding="utf-8")
        for line in cpu_info.splitlines():
            if line.lower().startswith(("model name", "hardware")):
                cpu = line.split(":", 1)[1].strip()
                break
    except (OSError, IndexError):
        pass
    devices = []
    output = command_output(["lspci"])
    for line in output.splitlines():
        if "vga" in line.lower() or "3d controller" in line.lower():
            devices.append({"name": line.split(": ", 1)[-1], "vendor": "unknown", "vram_mib": None})
    return memory, cpu, devices


def recommend_models(ram_gib: float | None, vram_mib: int | None, cores: int) -> dict[str, Any]:
    memory = ram_gib or 0
    vram = vram_mib or 0
    if vram >= 4096 or memory >= 12:
        recommended = "large-v3-turbo"
        reason = "Enough memory for the recommended multilingual turbo model; actual speed still depends on the backend."
    elif memory >= 6 and cores >= 6:
        recommended = "medium"
        reason = "A balanced CPU-oriented choice for the detected memory and processor."
    elif memory >= 3:
        recommended = "small"
        reason = "A conservative choice for limited memory."
    else:
        recommended = "base"
        reason = "A low-memory choice; expect lower recognition quality."

    names = ["small", "medium", "large-v3-turbo-q5_0", "large-v3-turbo", "large-v3"]
    options = []
    for index, name in enumerate(names, start=1):
        options.append({
            "number": index,
            "name": name,
            "disk_mib": MODELS[name]["disk_mib"],
            "recommended": name == recommended,
            "label": ("Fast draft", "Balanced CPU", "Compact large", "Recommended quality/speed", "Maximum quality")[index - 1],
        })
    return {"recommended": recommended, "reason": reason, "options": options}


def detect_hardware() -> dict[str, Any]:
    system = normalized_system()
    architecture = normalized_architecture()
    ram_bytes = None
    cpu_name = platform.processor() or None
    devices: list[dict[str, Any]] = []
    if system == "windows":
        ram_bytes, cpu_name, devices = windows_details()
    elif system == "macos":
        ram_bytes, cpu_name, devices = macos_details()
    elif system == "linux":
        ram_bytes, cpu_name, devices = linux_details()

    nvidia = nvidia_gpus()
    known_names = {item.get("name") for item in nvidia}
    devices = nvidia + [item for item in devices if item.get("name") not in known_names]
    cores = os.cpu_count() or 1
    ram_gib = round(ram_bytes / 1024**3, 1) if ram_bytes else None
    vram_values = [item["vram_mib"] for item in nvidia if item.get("vram_mib")]
    vram_mib = max(vram_values) if vram_values else None

    if system == "macos":
        backend = "metal"
    elif nvidia:
        backend = "cuda"
    elif shutil.which("vulkaninfo"):
        backend = "vulkan"
    else:
        backend = "cpu"

    tools = {
        "ffmpeg": ffmpeg_path(),
        "cmake": shutil.which("cmake"),
        "git": shutil.which("git"),
        "nvidia-smi": shutil.which("nvidia-smi"),
        "vulkaninfo": shutil.which("vulkaninfo"),
        "package_managers": {
            name: shutil.which(name) for name in ("winget", "brew", "apt-get", "dnf", "pacman")
        },
    }
    return {
        "system": system,
        "system_description": platform.platform(),
        "architecture": architecture,
        "cpu": {"name": cpu_name, "logical_threads": cores},
        "memory": {"ram_bytes": ram_bytes, "ram_gib": ram_gib},
        "gpus": devices,
        "gpu_detection_complete": bool(nvidia) or system == "macos" or not devices,
        "recommended_backend": backend,
        "model_recommendation": recommend_models(ram_gib, vram_mib, cores),
        "tools": tools,
        "notes": [
            "Model recommendations are conservative estimates, not performance benchmarks.",
            "If GPU memory is unknown, choose a smaller model or run a smoke test before a long recording.",
        ],
    }
