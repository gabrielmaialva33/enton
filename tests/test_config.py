import os
from pathlib import Path
from unittest.mock import patch

from enton.core.config import Settings


def test_config_defaults():
    """Test default values without environment variables or .env file."""
    # Ensure no env vars interfere and ignore .env file
    with patch.dict(os.environ, {}, clear=True):
        settings = Settings(_env_file=None)
        assert settings.camera_ip == "localhost"
        assert settings.camera_source == "0"
        
        # Check dynamic path generation
        home = Path.home()
        expected_blob_root = str(home / ".enton" / "blobs")
        assert settings.blob_store_root == expected_blob_root

def test_load_from_dot_env():
    """Test that values are loaded from the project's .env file."""
    # This assumes .env exists in the root and has CAMERA_IP set
    if not Path(".env").exists():
        return
        
    settings = Settings()
    # verify it read something from .env (specific to user's env)
    # The user's .env has CAMERA_IP=192.168.18.23
    assert settings.camera_ip == "192.168.18.23"

def test_config_env_override():
    """Test that environment variables override defaults."""
    mock_env = {
        "CAMERA_IP": "10.0.0.1",
        "BLOB_STORE_ROOT": "/tmp/enton_test_blobs"
    }
    with patch.dict(os.environ, mock_env, clear=True):
        settings = Settings()
        assert settings.camera_ip == "10.0.0.1"
        assert settings.blob_store_root == "/tmp/enton_test_blobs"
