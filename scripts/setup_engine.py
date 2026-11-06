from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_CONFIG,
    ENGINE_RELEASE,
    ENGINE_REPOSITORY,
    ENGINE_VERSION,
    MODELS,
    MODELS_DIR,
    RELEASE_ASSETS,
    RUNTIME_DIR,
    download_file,
    ensure_directories,
    ffmpeg_path,
    find_engine_binary,
    load_config,
    model_file,
    portable_path,
    safe_extract_archive,
    save_config,
)
from hardware import detect_hardware, normalized_architecture, normalized_system


def dependency_instructions(system: str) -> dict[str, str]:
    return {
        "windows": "FFmpeg: winget install Gyan.FFmpeg. Source builds require CMake and MSVC Build Tools.",
        "macos": "FFmpeg: brew install ffmpeg. Source builds require Xcode Command Line Tools and CMake.",
        "linux": "Install FFmpeg, CMake and a C++ compiler with your distribution package manager.",
    }.get(system, "Install FFmpeg; source builds also require CMake and a C++ compiler.")


def resolve_backend(requested: str, hardware: dict[str, Any], model_name: str) -> str:
    if requested == "auto":
        system = hardware["system"]
        architecture = hardware["architecture"]
        if model_name in {"tiny", "base", "small"} and (system, architecture, "cpu") in RELEASE_ASSETS:
            return "cpu"
        return hardware["recommended_backend"]
    if requested == "metal" and hardware["system"] != "macos":
        raise RuntimeError("Metal backend is available only on macOS.")
    return requested


def install_prebuilt(system: str, architecture: str, backend: str) -> Path | None:
    asset = RELEASE_ASSETS.get((system, architecture, backend))
    if not asset:
        return None
    name = asset["name"]
    url = f"{ENGINE_REPOSITORY}/releases/download/{ENGINE_RELEASE}/{name}"
    archive = RUNTIME_DIR / "downloads" / name
    download_file(url, archive, "sha256", asset["sha256"])
    target = RUNTIME_DIR / f"whisper-{ENGINE_RELEASE}-{system}-{architecture}-{backend}"
    if not find_binary(target):
        safe_extract_archive(archive, target)
    binary = find_binary(target)
    if not binary:
        raise RuntimeError(f"Archive {name} does not contain whisper-cli.")
    if os.name != "nt":
        binary.chmod(binary.stat().st_mode | 0o111)
    return binary


def find_binary(root: Path) -> Path | None:
    executable = "whisper-cli.exe" if os.name == "nt" else "whisper-cli"
    candidates = sorted(root.glob(f"**/{executable}"))
    return candidates[0] if candidates else None


def run_build(command: list[str]) -> None:
    result = subprocess.run(command, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Build command failed with exit code {result.returncode}: {' '.join(command)}")


def install_from_source(system: str, architecture: str, backend: str, threads: int) -> Path:
    if not shutil.which("cmake"):
        raise RuntimeError(f"CMake is required for this platform/backend. {dependency_instructions(system)}")
    source_archive = RUNTIME_DIR / "downloads" / f"whisper-{ENGINE_RELEASE}-source.zip"
    download_file(f"{ENGINE_REPOSITORY}/archive/refs/tags/{ENGINE_RELEASE}.zip", source_archive)
    source_root = RUNTIME_DIR / f"source-{ENGINE_RELEASE}"
    if not (source_root / "CMakeLists.txt").exists():
        safe_extract_archive(source_archive, source_root)
        directories = [item for item in source_root.iterdir() if item.is_dir()]
        if len(directories) == 1:
            for item in directories[0].iterdir():
                shutil.move(str(item), str(source_root / item.name))
            directories[0].rmdir()
    build_dir = source_root / "build"
    configure = [
        "cmake", "-S", str(source_root), "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=OFF",
    ]
    flags = {
        "cuda": ["-DGGML_CUDA=ON"],
        "vulkan": ["-DGGML_VULKAN=ON"],
        "metal": ["-DGGML_METAL=ON"],
        "cpu": ["-DGGML_CUDA=OFF", "-DGGML_VULKAN=OFF", "-DGGML_METAL=OFF"],
    }
    run_build(configure + flags[backend])
    run_build(["cmake", "--build", str(build_dir), "--config", "Release", "--parallel", str(threads)])
    binary = find_binary(build_dir)
    if not binary:
        raise RuntimeError("Build completed but whisper-cli was not found.")
    if os.name != "nt":
        binary.chmod(binary.stat().st_mode | 0o111)
    return binary


def install_model(name: str) -> Path:
    if name not in MODELS:
        raise RuntimeError(f"Unknown model '{name}'. Available: {', '.join(MODELS)}")
    if ".en" in name or name.endswith("-tdrz"):
        raise RuntimeError("Choose a multilingual model without .en or -tdrz for general meetings.")
    target = MODELS_DIR / f"ggml-{name}.bin"
    metadata = MODELS[name]
    url = f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{name}.bin"
    return download_file(url, target, "sha1", metadata["sha1"])


def install_vad_model() -> Path:
    target = MODELS_DIR / "ggml-silero-v6.2.0.bin"
    url = "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin"
    return download_file(url, target)


def setup(
    model_name: str,
    backend: str = "auto",
    threads: int | None = None,
    language: str = "auto",
    summary_language: str = "ru",
    initial_prompt: str = "",
    vad: bool = True,
    source_build: bool = False,
    summary_enabled: bool = True,
    email_enabled: bool = False,
) -> dict[str, Any]:
    ensure_directories()
    if not ffmpeg_path():
        system = normalized_system()
        raise RuntimeError(f"FFmpeg was not found on PATH. {dependency_instructions(system)}")
    hardware = detect_hardware()
    selected_backend = resolve_backend(backend, hardware, model_name)
    threads = threads or min(hardware["cpu"]["logical_threads"], 8)
    if threads < 1:
        raise RuntimeError("Thread count must be positive.")

    warnings = []
    previous = load_config(required=False)
    binary = find_engine_binary(previous)
    same_backend = previous.get("engine", {}).get("backend") == selected_backend
    installation = "reused"
    if not binary or not same_backend or source_build:
        installation = "prebuilt"
        try:
            binary = None if source_build else install_prebuilt(
                hardware["system"], hardware["architecture"], selected_backend
            )
        except OSError as error:
            cpu_available = (hardware["system"], hardware["architecture"], "cpu") in RELEASE_ASSETS
            if backend != "auto" or selected_backend == "cpu" or not cpu_available:
                raise
            warnings.append(
                f"Не удалось скачать {selected_backend}-сборку ({error}). В режиме auto используется компактный CPU-пакет."
            )
            selected_backend = "cpu"
            binary = install_prebuilt(hardware["system"], hardware["architecture"], "cpu")
        if not binary and not source_build and backend == "auto" and selected_backend != "cpu":
            cpu_available = (hardware["system"], hardware["architecture"], "cpu") in RELEASE_ASSETS
            if cpu_available:
                warnings.append(
                    f"Для {selected_backend} нет предсобранного пакета. В режиме auto используется CPU без сборки из исходников."
                )
                selected_backend = "cpu"
                binary = install_prebuilt(hardware["system"], hardware["architecture"], "cpu")
        if not binary:
            binary = install_from_source(
                hardware["system"], hardware["architecture"], selected_backend, threads
            )
            installation = "source"

    model_path = install_model(model_name)
    vad_path = install_vad_model() if vad else None
    config = DEFAULT_CONFIG.copy()
    config = {
        **config,
        "engine": {
            "name": "whisper.cpp",
            "version": ENGINE_VERSION,
            "release": ENGINE_RELEASE,
            "backend": selected_backend,
            "requested_backend": backend,
            "binary": portable_path(binary),
            "installation": installation,
        },
        "model": {"name": model_name, "path": portable_path(model_path)},
        "artifacts": {"transcript": True, "summary": summary_enabled, "email": email_enabled},
        "transcription": {
            "language": language,
            "summary_language": summary_language,
            "threads": threads,
            "beam_size": 5,
            "vad": vad,
            "vad_model": portable_path(vad_path) if vad_path else "",
            "initial_prompt": initial_prompt,
        },
        "paths": {"recordings": "recordings", "outputs": "outputs", "logs": "logs"},
    }
    save_config(config)
    return {
        "success": True,
        "config": "config.json",
        "engine": config["engine"],
        "model": config["model"],
        "artifacts": config["artifacts"],
        "vad_model": config["transcription"]["vad_model"] or None,
        "ffmpeg": ffmpeg_path(),
        "warnings": warnings,
        "next_step": "Run scripts/meetings.py doctor for a real transcription smoke test.",
    }


def quick_check() -> dict[str, Any]:
    config = load_config()
    binary = find_engine_binary(config)
    model = model_file(config)
    errors = []
    if not binary:
        errors.append("whisper-cli binary is missing")
    if not model.is_file() or model.stat().st_size == 0:
        errors.append("Whisper model is missing or empty")
    if not ffmpeg_path():
        errors.append("ffmpeg is missing from PATH")
    vad_path = None
    if config["transcription"].get("vad"):
        vad_value = config["transcription"].get("vad_model") or ""
        if not vad_value:
            errors.append("VAD is enabled but no VAD model is configured")
        else:
            from common import resolve_path
            vad_path = resolve_path(vad_value)
            if not vad_path.is_file():
                errors.append("VAD model is missing")
    version = None
    if binary:
        result = subprocess.run(
            [str(binary), "--version"], capture_output=True, text=True,
            encoding="utf-8", errors="replace", check=False,
        )
        version = (result.stdout or result.stderr).strip()
        if result.returncode != 0:
            errors.append(f"whisper-cli --version failed with code {result.returncode}")
    return {
        "success": not errors,
        "engine_binary": portable_path(binary) if binary else None,
        "engine_version_output": version,
        "model": portable_path(model),
        "vad_model": portable_path(vad_path) if isinstance(vad_path, Path) and str(vad_path) else None,
        "ffmpeg": ffmpeg_path(),
        "errors": errors,
    }
