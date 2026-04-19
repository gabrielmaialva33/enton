# Enton Scripts

Utility scripts organized by function.

## Structure

- **`setup/`**: Initial setup and installation.
  - `phone_setup.sh`: Configures ADB and Android dependencies.
  - `load_commonsense.py`: Downloads and indexes common knowledge (ASCENT++).

- **`tests/`**: Verification and manual testing.
  - `smoke_test_f1.py`: Smoke test (starts the app and checks components).
  - `verify_phase2.py`: Validates the brain's thought loop.
  - `verify_phase3.py`: Validates dynamic skill registration.

- **`data/`**: Data and model management.
  - `optimize_models.py`: Converts YOLO models (`.pt`) to TensorRT (`.engine`).

- **`dev/`**: Development tools.
  - `live_yolo.py`: Real-time computer vision viewer.

## Usage

Run from the project root:

```bash
# Example: Run smoke test
uv run python scripts/tests/smoke_test_f1.py

# Example: Optimize models
uv run python scripts/data/optimize_models.py