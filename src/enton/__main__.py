from __future__ import annotations

import asyncio
import logging
import os


def main() -> None:
    # Prevent PyTorch/OpenBLAS multi-threading heap corruption
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

    import torch

    torch.set_num_threads(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from enton.app import App

    app = App()
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        logging.info("Enton shutting down.")


if __name__ == "__main__":
    main()
