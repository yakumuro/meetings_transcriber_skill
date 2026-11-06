from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PACKAGE_ROOT / "config.json"
RECORDINGS_DIR = PACKAGE_ROOT / "recordings"
OUTPUTS_DIR = PACKAGE_ROOT / "outputs"
MODELS_DIR = PACKAGE_ROOT / "models"
RUNTIME_DIR = PACKAGE_ROOT / "runtime"
LOGS_DIR = PACKAGE_ROOT / "logs"
STATE_DIR = PACKAGE_ROOT / "state"
REGISTRY_PATH = OUTPUTS_DIR / "REGISTRY.md"

ENGINE_VERSION = "v1.9.5"
ENGINE_RELEASE = "b5454"
ENGINE_REPOSITORY = "https://github.com/ggml-org/whisper.cpp"

MODELS: dict[str, dict[str, Any]] = {
    "tiny": {"disk_mib": 75, "sha1": "bd577a113a864445d4c299885e0cb97d4ba92b5f"},
    "base": {"disk_mib": 142, "sha1": "465707469ff3a37a2b9b8d8f89f2f99de7299dac"},
    "small": {"disk_mib": 466, "sha1": "55356645c2b361a969dfd0ef2c5a50d530afd8d5"},
    "medium": {"disk_mib": 1536, "sha1": "fd9727b6e1217c2f614f9b698455c4ffd82463b4"},
    "large-v3": {"disk_mib": 2969, "sha1": "ad82bf6a9043ceed055076d0fd39f5f186ff8062"},
    "large-v3-turbo": {"disk_mib": 1536, "sha1": "4af2b29d7ec73d781377bfd1758ca957a807e941"},
    "large-v3-turbo-q5_0": {"disk_mib": 547, "sha1": "e050f7970618a659205450ad97eb95a18d69c9ee"},
}

RELEASE_ASSETS: dict[tuple[str, str, str], dict[str, str]] = {
    ("windows", "x64", "cpu"): {
        "name": "whisper-bin-x64.zip",
        "sha256": "6ba69e3482d7826214f90a6a9c84ca07782aec1e1d0c6a7c30c994fd5d816ccb",
    },
    ("windows", "x64", "cuda"): {
        "name": "whisper-bin-win-cuda-12.4.0-x64.zip",
        "sha256": "afef0b881c500958921c3f5523b50e59ee2ec9b6f5cbd25b324c51ed308a957a",
    },
    ("windows", "arm64", "cpu"): {
        "name": "whisper-bin-win-cpu-arm64.zip",
        "sha256": "28c37e7b598c3d9bbfef94f3bd67f2ee6f12b7c86f3da5edcf5e4308d615ff6d",
    },
    ("linux", "x64", "cpu"): {
        "name": "whisper-bin-ubuntu-x64.tar.gz",
        "sha256": "a72becf15d7917f990f6313867a52638b82b7f9ef237fb0c980dac56a135781c",
    },
    ("linux", "arm64", "cpu"): {
        "name": "whisper-bin-ubuntu-arm64.tar.gz",
        "sha256": "6b95ebfc60447df48e70ef00a73bdc3f41679ed2d01ef465827206a2ff325149",
    },
}

DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "engine": {
        "name": "whisper.cpp",
        "version": ENGINE_VERSION,
        "release": ENGINE_RELEASE,
        "backend": "cpu",
        "binary": "",
    },
    "model": {"name": "large-v3-turbo", "path": ""},
    "artifacts": {"transcript": True, "summary": True, "email": False},
    "transcription": {
        "language": "auto",
        "summary_language": "ru",
        "threads": max(1, min(os.cpu_count() or 4, 8)),
        "beam_size": 5,
        "vad": True,
        "vad_model": "models/ggml-silero-v6.2.0.bin",
        "initial_prompt": "",
    },
    "paths": {
        "recordings": "recordings",
        "outputs": "outputs",
        "logs": "logs",
    },
}

SUPPORTED_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".aac",
    ".mp4", ".mkv", ".avi", ".mov", ".wmv",
}


def ensure_directories() -> None:
    for directory in (RECORDINGS_DIR, OUTPUTS_DIR, MODELS_DIR, RUNTIME_DIR, LOGS_DIR, STATE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PACKAGE_ROOT / path).resolve()


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PACKAGE_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def load_config(required: bool = True) -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        if required:
            raise RuntimeError("Configuration not found. Run Start or scripts/meetings.py setup first.")
        return json.loads(json.dumps(DEFAULT_CONFIG))
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read config.json: {error}") from error


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_config(config: dict[str, Any]) -> None:
    atomic_write_text(CONFIG_PATH, json.dumps(config, ensure_ascii=False, indent=2) + "\n")


def emit(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, target: Path, algorithm: str | None = None, expected: str | None = None) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        if algorithm and expected and file_digest(target, algorithm).lower() != expected.lower():
            raise RuntimeError(f"Existing file has an invalid {algorithm} checksum: {target}")
        return target

    temporary = target.with_name(target.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "meetings-transcriber-skill/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            total = int(response.headers.get("Content-Length", "0") or 0)
            downloaded = 0
            last_percent = -10
            if total:
                print(f"Downloading {target.name}: {total / 1024 / 1024:.1f} MiB", flush=True)
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                downloaded += len(chunk)
                if total:
                    percent = int(downloaded * 100 / total)
                    if percent >= last_percent + 10 or downloaded == total:
                        print(f"Downloading {target.name}: {percent}%", flush=True)
                        last_percent = percent
        if total:
            print()
        if temporary.stat().st_size == 0:
            raise RuntimeError(f"Downloaded file is empty: {url}")
        if algorithm and expected:
            actual = file_digest(temporary, algorithm)
            if actual.lower() != expected.lower():
                raise RuntimeError(f"Checksum mismatch for {target.name}: expected {expected}, got {actual}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def safe_extract_archive(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    root = target.resolve()
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                destination = (target / member.filename).resolve()
                if root != destination and root not in destination.parents:
                    raise RuntimeError(f"Unsafe archive member: {member.filename}")
            bundle.extractall(target)
        return
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            destination = (target / member.name).resolve()
            if root != destination and root not in destination.parents:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
        try:
            bundle.extractall(target, filter="data")
        except TypeError:
            if any(not (member.isfile() or member.isdir()) for member in bundle.getmembers()):
                raise RuntimeError("Tar links and special files require Python with safe extraction filters.")
            bundle.extractall(target)


def find_engine_binary(config: dict[str, Any] | None = None) -> Path | None:
    if config:
        configured = config.get("engine", {}).get("binary")
        if configured:
            candidate = resolve_path(configured)
            return candidate if candidate.is_file() else None
    executable = "whisper-cli.exe" if os.name == "nt" else "whisper-cli"
    direct = RUNTIME_DIR / executable
    if direct.is_file():
        return direct
    candidates = sorted(RUNTIME_DIR.glob(f"**/{executable}"))
    return candidates[0] if candidates else None


def model_file(config: dict[str, Any]) -> Path:
    configured = config.get("model", {}).get("path")
    if configured:
        return resolve_path(configured)
    name = config.get("model", {}).get("name", "")
    return MODELS_DIR / f"ggml-{name}.bin"


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^\w\s-]", "", value, flags=re.UNICODE)
    value = re.sub(r"[\s_-]+", "-", value, flags=re.UNICODE)
    return value.strip("-") or "meeting"


def format_timestamp(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def source_key(path: Path) -> str:
    stat = path.stat()
    identity = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def source_modified(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone()


def temporary_directory(prefix: str) -> Path:
    temporary_root = RUNTIME_DIR / "tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=temporary_root))
