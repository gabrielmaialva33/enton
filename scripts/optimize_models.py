#!/usr/bin/env python3
"""Optimize YOLO models → TensorRT .engine for Enton.

Auto-discovers all .pt files in models/ and exports to TensorRT FP16 or INT8.
Enton Vision auto-detects .engine files at runtime — no config change needed.

Usage:
    uv run python scripts/optimize_models.py                # FP16 (default)
    uv run python scripts/optimize_models.py --int8           # INT8 (requires calibration data)
    uv run python scripts/optimize_models.py models/yolo11s.pt    # specific model
    uv run python scripts/optimize_models.py --force         # re-export even if .engine exists
    uv run python scripts/optimize_models.py --workspace 8   # 8GB TensorRT workspace
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Pathfeat: introduce i18n configuration, add `ContextEngine` tests, and update `.gitignore` to exclude build artifacts and sensitive files.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("optimize")

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_CALIBRATION_DATA = "coco128.yaml"


def optimize_model(
    path: Path,
    *,
    device: int = 0,
    half: bool = True,
    workspace: int = 4,
    force: bool = False,
    int8: bool = False,
    data: str | None = None,
) -> Path | None:
    """Export a single .pt model to TensorRT .engine.

    Returns the engine path on success, None on failure.
    """
    if not path.exists():
        logger.warning("Not found: %s", path)
        return None

    if path.suffix == ".engine":
        logger.info("Already an engine: %s", path.name)
        return path

    engine_path = path.with_suffix(".engine")
    if engine_path.exists() and not force:
        size_mb = engine_path.stat().st_size / 1024 / 1024
        logger.info("Engine exists: %s (%.1f MB) — skip (use --force to re-export)", engine_path.name, size_mb)
        return engine_path

    precision = "INT8" if int8 else "FP16" if half else "FP32"
    logger.info(
        "Exporting %s → TensorRT %s (device=%d, workspace=%dGB)...",
        path.name,
        precision,
        device,
        workspace,
    )
    t0 = time.monotonic()

    try:
        from ultralytics import YOLO

        model = YOLO(str(path))
        export_kwargs = {
            "format": "engine",
            "device": device,
            "simplify": True,
            "workspace": workspace,
            "half": half,
            "int8": int8,
        }
        if int8:
            export_kwargs["data"] = data
            export_kwargs["half"] = False  # INT8 and FP16 are mutually exclusive

        model.export(**export_kwargs)
        elapsed = time.monotonic() - t0
        size_mb = engine_path.stat().st_size / 1024 / 1024
        logger.info("Done: %s (%.1f MB) in %.1fs", engine_path.name, size_mb, elapsed)
        return engine_path
    except Exception:
        logger.exception("Failed to export %s", path.name)
        return None


def discover_models(models_dir: Path) -> list[Path]:
    """Find all .pt YOLO model files in the models directory."""
    if not models_dir.exists():
        return []
    return sorted(models_dir.glob("*.pt"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize YOLO models to TensorRT")
    parser.add_argument("models", nargs="*", help="Specific .pt files (default: auto-discover models/)")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index (default: 0)")
    parser.add_argument("--workspace", type=int, default=4, help="TensorRT workspace in GB (default: 4)")
    parser.add_argument("--force", action="store_true", help="Re-export even if .engine exists")
    parser.add_argument("--no-half", action="store_true", help="Disable FP16 (use FP32)")
    parser.add_argument("--int8", action="store_true", help="Enable INT8 quantization")
    parser.add_argument(
        "--data",
        type=str,
        default=DEFAULT_CALIBRATION_DATA,
        help=f"Calibration data for INT8. Defaults to {DEFAULT_CALIBRATION_DATA}",
    )
    args = parser.parse_args()

    if args.int8 and args.no_half:
        logger.error("Cannot use --int8 and --no-half (FP32) at the same time.")
        return

    if args.models:
        paths = [Path(m) for m in args.models]
    else:
        paths = discover_models(MODELS_DIR)

    if not paths:
        logger.warning("No .pt models found. Place YOLO .pt files in %s", MODELS_DIR)
        return

    logger.info("Found %d model(s) to optimize", len(paths))
    success = 0
    for p in paths:
        result = optimize_model(
            p,
            device=args.device,
            half=not args.no_half and not args.int8,
            workspace=args.workspace,
            force=args.force,
            int8=args.int8,
            data=args.data if args.int8 else None,
        )
        if result is not None:
            success += 1

    logger.info("Optimized %d/%d models", success, len(paths))


if __name__ == "__main__":
    main()
