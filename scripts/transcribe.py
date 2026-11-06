from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from common import (
    LOGS_DIR,
    OUTPUTS_DIR,
    PACKAGE_ROOT,
    RECORDINGS_DIR,
    RUNTIME_DIR,
    STATE_DIR,
    SUPPORTED_EXTENSIONS,
    atomic_write_text,
    download_file,
    ensure_directories,
    ffmpeg_path,
    find_engine_binary,
    format_timestamp,
    load_config,
    model_file,
    portable_path,
    resolve_path,
    slugify,
    source_key,
    source_modified,
    temporary_directory,
)
from setup_engine import quick_check

_ACTIVE_CONFIG = load_config(required=False)
RECORDINGS_DIR = resolve_path(_ACTIVE_CONFIG.get("paths", {}).get("recordings", "recordings"))
OUTPUTS_DIR = resolve_path(_ACTIVE_CONFIG.get("paths", {}).get("outputs", "outputs"))
LOGS_DIR = resolve_path(_ACTIVE_CONFIG.get("paths", {}).get("logs", "logs"))

CURRENT_JOB = STATE_DIR / "current-job.json"


def read_job() -> dict[str, Any] | None:
    if not CURRENT_JOB.exists():
        return None
    try:
        return json.loads(CURRENT_JOB.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_job(job: dict[str, Any]) -> None:
    atomic_write_text(CURRENT_JOB, json.dumps(job, ensure_ascii=False, indent=2) + "\n")


def process_exists(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
        )
        return str(pid) in result.stdout
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def get_status() -> dict[str, Any]:
    job = read_job()
    if not job:
        return {"running": False, "status": "idle", "job": None}
    active = job.get("status") in {"starting", "running"}
    alive = process_exists(job.get("pid"))
    if active and alive:
        return {**job, "running": True}
    if active and not alive:
        job = {
            **job,
            "status": "failed",
            "running": False,
            "error": "The transcription process is no longer running. Inspect the job logs.",
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        write_job(job)
    return {"running": False, **job}


def resolve_input(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        candidate = candidate.resolve()
    else:
        by_name = RECORDINGS_DIR / candidate.name
        candidate = by_name.resolve() if by_name.is_file() else (PACKAGE_ROOT / candidate).resolve()
    if not candidate.is_file():
        raise RuntimeError(f"Recording not found: {candidate}")
    if candidate.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise RuntimeError(f"Unsupported file type: {candidate.suffix}")
    return candidate


def processed_sources() -> dict[str, str]:
    processed = {}
    if not OUTPUTS_DIR.exists():
        return processed
    for metadata_path in OUTPUTS_DIR.glob("*/metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            key = metadata.get("source", {}).get("key")
            if key and metadata.get("status") == "completed":
                processed[key] = metadata_path.parent.name
        except (OSError, json.JSONDecodeError):
            continue
    return processed


def recording_record(path: Path, processed: dict[str, str]) -> dict[str, Any]:
    stat = path.stat()
    key = source_key(path)
    return {
        "id": path.stem,
        "filename": path.name,
        "path": str(path.resolve()),
        "relative_path": portable_path(path),
        "modified_at": source_modified(path).isoformat(timespec="seconds"),
        "modified_date": source_modified(path).date().isoformat(),
        "size_bytes": stat.st_size,
        "source_key": key,
        "already_transcribed": key in processed,
        "output_folder": processed.get(key),
    }


def list_recordings(days: int = 1, include_all: bool = False) -> dict[str, Any]:
    if days < 1:
        raise RuntimeError("--days must be at least 1")
    files = []
    if RECORDINGS_DIR.exists():
        files = [
            path for path in RECORDINGS_DIR.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    files.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    if not include_all:
        earliest = date.today() - timedelta(days=days - 1)
        files = [path for path in files if earliest <= source_modified(path).date() <= date.today()]
    processed = processed_sources()
    records = [recording_record(path, processed) for path in files]
    return {
        "recordings_dir": str(RECORDINGS_DIR),
        "outputs_dir": str(OUTPUTS_DIR),
        "filter_days": None if include_all else days,
        "count": len(records),
        "latest_recording": records[0] if records else None,
        "recordings": records,
    }


def build_output_dir(source: Path, output_name: str | None) -> Path:
    if output_name:
        folder = slugify(output_name)
    else:
        folder = f"{source_modified(source).strftime('%Y-%m-%d')}_{slugify(source.stem)}"
    target = OUTPUTS_DIR / folder
    if target.exists():
        raise RuntimeError(
            f"Output folder already exists: {target}. Choose another --output-name; existing results are not overwritten."
        )
    return target


def extract_audio(source: Path, target: Path) -> None:
    ffmpeg = ffmpeg_path()
    if not ffmpeg:
        raise RuntimeError("FFmpeg was not found on PATH.")
    command = [
        ffmpeg, "-nostdin", "-y", "-i", str(source), "-vn",
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(target),
    ]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed with code {result.returncode}:\n{result.stderr[-4000:]}")


def cli_file_path(path: Path, working_dir: Path) -> str:
    path = path.resolve()
    if os.name != "nt":
        return str(path)
    try:
        relative = os.path.relpath(path, working_dir.resolve())
        relative.encode("ascii")
        return relative
    except (ValueError, UnicodeEncodeError):
        pass
    try:
        str(path).encode("ascii")
        return str(path)
    except UnicodeEncodeError:
        pass
    try:
        import ctypes
        buffer = ctypes.create_unicode_buffer(32768)
        length = ctypes.windll.kernel32.GetShortPathNameW(str(path), buffer, len(buffer))
        short = buffer.value if length else ""
        if not short:
            raise OSError("Short path is unavailable")
        short.encode("ascii")
        return short
    except (OSError, UnicodeEncodeError):
        pass
    stat = path.stat()
    identity = f"{path}|{stat.st_size}|{stat.st_mtime_ns}"
    target = RUNTIME_DIR / "cli-files" / (uuid.uuid5(uuid.NAMESPACE_URL, identity).hex + path.suffix)
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(path, target)
        except OSError:
            shutil.copy2(path, target)
    return os.path.relpath(target, working_dir.resolve())


def execute_whisper(audio: Path, config: dict[str, Any], job_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    binary = find_engine_binary(config)
    if not binary:
        raise RuntimeError("whisper-cli is not installed. Run setup first.")
    model = model_file(config)
    if not model.is_file():
        raise RuntimeError(f"Model not found: {model}")
    settings = config["transcription"]
    raw_output = audio.with_name(audio.stem + "-whisper")
    command = [
        str(binary), "--model", cli_file_path(model, binary.parent),
        "--file", cli_file_path(audio, binary.parent),
        "--language", settings.get("language", "auto"),
        "--threads", str(settings.get("threads", 4)),
        "--beam-size", str(settings.get("beam_size", 5)),
        "--output-json", "--output-file", cli_file_path(raw_output, binary.parent),
    ]
    if settings.get("initial_prompt"):
        command.extend(["--prompt", settings["initial_prompt"]])
    if config.get("engine", {}).get("backend") == "cpu":
        command.append("--no-gpu")
    if settings.get("vad"):
        vad_model = resolve_path(settings.get("vad_model", ""))
        if not vad_model.is_file():
            raise RuntimeError(f"VAD model not found: {vad_model}")
        command.extend(["--vad", "--vad-model", cli_file_path(vad_model, binary.parent)])
    result = subprocess.run(
        command, cwd=str(binary.parent), capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
    )
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        LOGS_DIR / f"engine-{job_id}.log",
        f"COMMAND: {' '.join(command)}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}",
    )
    json_path = raw_output.with_suffix(".json")
    if result.returncode != 0:
        raise RuntimeError(
            f"whisper-cli failed with code {result.returncode}. See logs/engine-{job_id}.log.\n"
            f"{result.stderr[-2000:]}"
        )
    if not json_path.is_file():
        raise RuntimeError("whisper-cli completed without producing JSON output.")
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Cannot parse whisper-cli JSON: {error}") from error
    return payload, {
        "stdout_tail": result.stdout[-1000:],
        "stderr_tail": result.stderr[-1000:],
        "engine_log": portable_path(LOGS_DIR / f"engine-{job_id}.log"),
    }


def parse_segments(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_segments = payload.get("transcription") or payload.get("segments") or []
    segments = []
    for item in raw_segments:
        offsets = item.get("offsets") or {}
        start = float(offsets.get("from", item.get("start", 0))) / 1000
        end = float(offsets.get("to", item.get("end", 0))) / 1000
        text = str(item.get("text", "")).strip()
        if text:
            segments.append({"start": start, "end": end, "text": text})
    return segments


def probe_duration(source: Path, segments: list[dict[str, Any]]) -> float:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        result = subprocess.run([
            ffprobe, "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(source),
        ], capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
        try:
            return round(float(result.stdout.strip()), 2)
        except ValueError:
            pass
    return round(max((segment["end"] for segment in segments), default=0), 2)


def write_results(
    source: Path,
    output_dir: Path,
    payload: dict[str, Any],
    segments: list[dict[str, Any]],
    config: dict[str, Any],
    engine_info: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    language = (payload.get("result") or {}).get("language") or config["transcription"].get("language")
    duration = probe_duration(source, segments)
    lines = [
        f"# Transcript: {source.name}", "", "## Metadata", "",
        f"- **Source file:** `{source.name}`",
        f"- **Recording date:** {source_modified(source).strftime('%Y-%m-%d %H:%M')}",
        f"- **Model:** {config['model']['name']}",
        f"- **Engine:** whisper.cpp {config['engine']['version']} ({config['engine']['backend']})",
        f"- **Language:** {language}", "", "## Full transcription", "",
    ]
    for segment in segments:
        lines.append(f"**[{format_timestamp(segment['start'])} - {format_timestamp(segment['end'])}]** {segment['text']}")
        lines.append("")
    transcript = output_dir / "transcript.md"
    atomic_write_text(transcript, "\n".join(lines))
    metadata = {
        "schema_version": 1,
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "filename": source.name,
            "relative_path": portable_path(source),
            "key": source_key(source),
            "modified_at": source_modified(source).isoformat(timespec="seconds"),
            "size_bytes": source.stat().st_size,
        },
        "engine": config["engine"],
        "model": config["model"],
        "language": language,
        "duration_seconds": duration,
        "segments_count": len(segments),
        "transcript": "transcript.md",
        "engine_log": engine_info["engine_log"],
    }
    atomic_write_text(output_dir / "metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    return {
        "output_dir": str(output_dir),
        "output_folder": output_dir.name,
        "transcript": str(transcript),
        "metadata": str(output_dir / "metadata.json"),
        "language": language,
        "duration_seconds": duration,
        "segments_count": len(segments),
    }


def transcribe_file(raw_input: str, output_name: str | None = None, job_id: str | None = None) -> dict[str, Any]:
    ensure_directories()
    config = load_config()
    source = resolve_input(raw_input)
    key = source_key(source)
    processed = processed_sources()
    if key in processed:
        raise RuntimeError(f"Recording was already transcribed into outputs/{processed[key]}. Refusing to overwrite it.")
    output_dir = build_output_dir(source, output_name)
    job_id = job_id or uuid.uuid4().hex[:12]
    temp_dir = temporary_directory("meetings-transcriber-")
    try:
        audio = temp_dir / "audio.wav"
        extract_audio(source, audio)
        payload, engine_info = execute_whisper(audio, config, job_id)
        segments = parse_segments(payload)
        if not segments:
            raise RuntimeError("Transcription completed but no speech segments were returned; no result was indexed.")
        result = write_results(source, output_dir, payload, segments, config, engine_info)
        return {"success": True, "job_id": job_id, "input": str(source), **result}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_job(raw_input: str, output_name: str | None, job_id: str) -> dict[str, Any]:
    job = read_job() or {"job_id": job_id}
    job.update({
        "job_id": job_id,
        "status": "running",
        "pid": os.getpid(),
        "input": raw_input,
        "started_at": job.get("started_at") or datetime.now(timezone.utc).isoformat(),
    })
    write_job(job)
    try:
        result = transcribe_file(raw_input, output_name=output_name, job_id=job_id)
        job.update({"status": "completed", "result": result, "finished_at": datetime.now(timezone.utc).isoformat()})
        write_job(job)
        return result
    except Exception as error:
        job.update({
            "status": "failed", "error": str(error),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        })
        write_job(job)
        raise


def start_background(raw_input: str, output_name: str | None) -> dict[str, Any]:
    ensure_directories()
    source = resolve_input(raw_input)
    status = get_status()
    if status.get("running"):
        raise RuntimeError(f"Transcription job {status.get('job_id')} is already running with pid {status.get('pid')}.")
    key = source_key(source)
    processed = processed_sources()
    if key in processed:
        raise RuntimeError(f"Recording was already transcribed into outputs/{processed[key]}.")
    if output_name and (OUTPUTS_DIR / slugify(output_name)).exists():
        raise RuntimeError("The requested output folder already exists.")
    if not output_name:
        build_output_dir(source, None)

    load_config()
    job_id = uuid.uuid4().hex[:12]
    stdout_path = LOGS_DIR / f"job-{job_id}.log"
    stderr_path = LOGS_DIR / f"job-{job_id}-err.log"
    command = [
        sys.executable, str(PACKAGE_ROOT / "scripts" / "meetings.py"),
        "transcribe", str(source), "--foreground", "--job-id", job_id,
    ]
    if output_name:
        command.extend(["--output-name", output_name])
    job = {
        "job_id": job_id,
        "status": "starting",
        "input": str(source),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        options: dict[str, Any] = {
            "cwd": str(PACKAGE_ROOT), "stdin": subprocess.DEVNULL,
            "stdout": stdout, "stderr": stderr, "close_fds": True,
        }
        if os.name == "nt":
            options["creationflags"] = 0x00000200 | 0x08000000
        else:
            options["start_new_session"] = True
        process = subprocess.Popen(command, **options)
    job["pid"] = process.pid
    write_job(job)
    for _ in range(50):
        time.sleep(0.1)
        current = read_job() or job
        if current.get("status") == "running":
            break
        if process.poll() is not None:
            raise RuntimeError(f"Background job exited early with code {process.returncode}. See {stderr_path.name}.")
    return {
        "started": True,
        "job_id": job_id,
        "pid": process.pid,
        "input": str(source),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }


def doctor(quick: bool = False) -> dict[str, Any]:
    checks = quick_check()
    if quick or not checks["success"]:
        return checks
    sample = RUNTIME_DIR / "samples" / "jfk.wav"
    download_file(
        f"https://raw.githubusercontent.com/ggml-org/whisper.cpp/{ENGINE_RELEASE_FOR_SAMPLE}/samples/jfk.wav",
        sample,
    )
    config = load_config()
    job_id = "doctor-" + uuid.uuid4().hex[:8]
    try:
        payload, engine_info = execute_whisper(sample, config, job_id)
        segments = parse_segments(payload)
        text = " ".join(segment["text"] for segment in segments).strip()
        if not text:
            raise RuntimeError("Smoke test returned an empty transcript.")
        checks.update({
            "success": True,
            "smoke_test": "passed",
            "sample_text": text,
            "segments_count": len(segments),
            "engine_log": engine_info["engine_log"],
        })
    except Exception as error:
        checks.update({"success": False, "smoke_test": "failed", "errors": checks["errors"] + [str(error)]})
    return checks


ENGINE_RELEASE_FOR_SAMPLE = "b5454"
