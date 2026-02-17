#!/usr/bin/env python3
"""Optimize YOLO models → TensorRT .engine for Enton.

Auto-discovers all .pt files in models/ and exports to TensorRT FP16.
Enton Vision auto-detects .engine files at runtime — no config change needed.

Usage:
    uv run python scripts/optimize_models.py              # all models
    uv run python scripts/optimize_models.py models/yolo11s.pt  # specific model
    uv run python scripts/optimize_models.py --force       # re-export even if .engine exists
    uv run python scripts/optimize_models.py --workspace 8 # 8GB TensorRT workspace
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("optimize")

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def optimize_model(
    path: Path,
    *,
    device: int = 0,
    half: bool = True,
    workspace: int = 4,
    force: bool = False,
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

    logger.info("Exporting %s → TensorRT FP16 (device=%d, workspace=%dGB)...", path.name, device, workspace)
    t0 = time.monotonic()

    try:
        from ultralytics import YOLO

        model = YOLO(str(path))
        model.export(
            format="engine",
            device=device,
            half=half,
            simplify=True,
            workspace=workspace,
        )
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
    args = parser.parse_args()

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
            half=not args.no_half,
            workspace=args.workspace,
            force=args.force,
        )
        if result is not None:
            success += 1

    logger.info("Optimized %d/%d models", success, len(paths))


if __name__ == "__main__":
    main()
