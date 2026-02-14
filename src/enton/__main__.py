from __future__ import annotations

import asyncio
import logging


def main() -> None:
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
