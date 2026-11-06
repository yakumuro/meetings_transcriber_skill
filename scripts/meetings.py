#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from typing import Any

from common import CONFIG_PATH, MODELS, emit, load_config
from hardware import detect_hardware
from registry import add_entry, search_entries
from setup_engine import setup
from transcribe import doctor, get_status, list_recordings, run_job, start_background, transcribe_file


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Local meeting transcription with whisper.cpp")
    subparsers = result.add_subparsers(dest="command", required=True)

    subparsers.add_parser("detect", help="Detect hardware and recommend a model (JSON)")

    setup_parser = subparsers.add_parser("setup", help="Install whisper.cpp and a model")
    setup_parser.add_argument("--model", choices=list(MODELS))
    setup_parser.add_argument("--backend", choices=["auto", "cpu", "metal", "cuda", "vulkan"], default="auto")
    setup_parser.add_argument("--threads", type=int)
    setup_parser.add_argument("--language", default="auto")
    setup_parser.add_argument("--summary-language", default="ru")
    setup_parser.add_argument("--initial-prompt", default="")
    setup_parser.add_argument("--no-vad", action="store_true")
    setup_parser.add_argument("--source-build", action="store_true")
    setup_parser.add_argument("--summary", action=argparse.BooleanOptionalAction, default=True)
    setup_parser.add_argument("--email", action=argparse.BooleanOptionalAction, default=False)

    doctor_parser = subparsers.add_parser("doctor", help="Check installation and transcribe a sample")
    doctor_parser.add_argument("--quick", action="store_true")

    subparsers.add_parser("config", help="Show the active local configuration")

    list_parser = subparsers.add_parser("list", help="List recordings")
    list_parser.add_argument("--days", type=int, default=1)
    list_parser.add_argument("--all", action="store_true")

    transcribe_parser = subparsers.add_parser("transcribe", help="Transcribe a recording")
    transcribe_parser.add_argument("input")
    transcribe_parser.add_argument("--background", action="store_true")
    transcribe_parser.add_argument("--foreground", action="store_true", help=argparse.SUPPRESS)
    transcribe_parser.add_argument("--output-name")
    transcribe_parser.add_argument("--job-id", help=argparse.SUPPRESS)

    subparsers.add_parser("status", help="Show current transcription job")

    registry_parser = subparsers.add_parser("registry", help="Update or search REGISTRY.md")
    registry_subparsers = registry_parser.add_subparsers(dest="registry_command", required=True)
    add_parser = registry_subparsers.add_parser("add")
    add_parser.add_argument("--folder", required=True)
    add_parser.add_argument("--date", required=True)
    add_parser.add_argument("--topics", required=True)
    add_parser.add_argument("--decisions", required=True)
    search_parser = registry_subparsers.add_parser("search")
    search_parser.add_argument("query", nargs="?", default="")
    search_parser.add_argument("--limit", type=int, default=20)
    return result


def dispatch(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "detect":
        return detect_hardware()
    if args.command == "setup":
        hardware = detect_hardware()
        model = args.model or hardware["model_recommendation"]["recommended"]
        return setup(
            model_name=model, backend=args.backend, threads=args.threads,
            language=args.language, summary_language=args.summary_language,
            initial_prompt=args.initial_prompt, vad=not args.no_vad,
            source_build=args.source_build,
            summary_enabled=args.summary, email_enabled=args.email,
        )
    if args.command == "doctor":
        return doctor(quick=args.quick)
    if args.command == "config":
        return {"path": str(CONFIG_PATH), "config": load_config()}
    if args.command == "list":
        return list_recordings(days=args.days, include_all=args.all)
    if args.command == "transcribe":
        if args.background:
            return start_background(args.input, args.output_name)
        if args.job_id:
            return run_job(args.input, args.output_name, args.job_id)
        return transcribe_file(args.input, output_name=args.output_name)
    if args.command == "status":
        return get_status()
    if args.command == "registry" and args.registry_command == "add":
        return add_entry(args.folder, args.date, args.topics, args.decisions)
    if args.command == "registry" and args.registry_command == "search":
        return search_entries(args.query, args.limit)
    raise RuntimeError("Unsupported command")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parser().parse_args()
    try:
        emit(dispatch(args))
        return 0
    except Exception as error:
        emit({"success": False, "error": str(error)})
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
