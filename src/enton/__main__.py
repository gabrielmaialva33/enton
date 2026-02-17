from __future__ import annotations

import argparse
import asyncio
import logging
import os

import torch

try:
    import uvloop
except ImportError:
    uvloop = None


def main() -> None:
    if uvloop:
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        
    parser = argparse.ArgumentParser(description="Enton — AI Robot Assistant")
    parser.add_argument("--webcam", action="store_true", help="Use local webcam instead of RTSP")
    parser.add_argument(
        "--viewer", action="store_true",
        help="Open live vision window with HUD overlay",
    )
    args = parser.parse_args()


    # Force webcam if requested (before Settings reads .env)
    if args.webcam:
        os.environ["CAMERA_SOURCE"] = "0"

    # Prevent PyTorch/OpenBLAS multi-threading heap corruption
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")


    torch.set_num_threads(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from enton.app import App
    app = App(viewer=args.viewer)
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        logger = logging.getLogger(__name__)
        logger.info("Enton shutting down.")


if __name__ == "__main__":
    main()
