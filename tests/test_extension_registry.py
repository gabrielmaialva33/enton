"""Tests for ExtensionRegistry."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from agno.tools import Toolkit

from enton.core.extension_registry import (
    ExtensionMeta,
    ExtensionRegistry,
    ExtensionSource,
    ExtensionState,
)


@pytest.fixture
def brain():
    return MagicMock()


@pytest.fixture
def registry(brain, tmp_path):
    return ExtensionRegistry(brain=brain, extensions_dir=str(tmp_path / "exts"))


class TestExtensionMeta:
    def test_summary(self):
        meta = ExtensionMeta(
            name="test_ext",
            source=ExtensionSource.BUILTIN,
            state=ExtensionState.ENABLED,
            tool_count=5,
        )
        s = meta.summary()
        assert "test_ext" in s
        assert "enabled" in s
        assert "5 tools" in s

    def test_success_rate_default(self):
        meta = ExtensionMeta(name="x", source=ExtensionSource.BUILTIN)
        assert meta.success_rate == 1.0

    def test_success_rate_with_errors(self):
        meta = ExtensionMeta(name="x", source=ExtensionSource.BUILTIN, calls=7, errors=3)
        assert meta.success_rate == 0.7


class TestExtensionRegistry:
    def test_register_builtin(self, registry):
        tk = Toolkit(name="my_tool")
        registry.register_builtin("my_tool", tk)

        meta = registry.get("my_tool")
        assert meta is not None
        assert meta.source == ExtensionSource.BUILTIN
        assert meta.state == ExtensionState.ENABLED

    def test_list_extensions(self, registry):
        tk1 = Toolkit(name="a")
        tk2 = Toolkit(name="b")
        registry.register_builtin("alpha", tk1)
        registry.register_builtin("beta", tk2)

        exts = registry.list_extensions()
        assert len(exts) == 2
        assert exts[0].name == "alpha"  # sorted

    def test_list_by_state(self, registry):
        tk = Toolkit(name="a")
        registry.register_builtin("a", tk)

        enabled = registry.list_extensions(state=ExtensionState.ENABLED)
        assert len(enabled) == 1

        disabled = registry.list_extensions(state=ExtensionState.DISABLED)
        assert len(disabled) == 0

    def test_discover_manifests(self, registry, tmp_path):
        ext_dir = tmp_path / "exts" / "my_ext"
        ext_dir.mkdir(parents=True)

        # Write manifest
        import json
        (ext_dir / "manifest.json").write_text(json.dumps({
            "name": "my_ext",
            "version": "1.0",
            "description": "Test",
            "module": "toolkit.py",
        }))

        # Write dummy toolkit
        (ext_dir / "toolkit.py").write_text(
            "from agno.tools import Toolkit\n"
            "class MyToolkit(Toolkit):\n"
            "    def __init__(self):\n"
            "        super().__init__(name='my_ext')\n"
        )

        found = registry.discover_manifests()
        assert "my_ext" in found
        assert registry.get("my_ext") is not None

    def test_load_from_manifest(self, registry, tmp_path):
        ext_dir = tmp_path / "exts" / "loadable"
        ext_dir.mkdir(parents=True)

        import json
        (ext_dir / "manifest.json").write_text(json.dumps({
            "name": "loadable",
            "module": "toolkit.py",
        }))
        (ext_dir / "toolkit.py").write_text(
            "from agno.tools import Toolkit\n"
            "class LoadableTools(Toolkit):\n"
            "    def __init__(self):\n"
            "        super().__init__(name='loadable')\n"
        )

        registry.discover_manifests()
        assert registry.load("loadable")
        meta = registry.get("loadable")
        assert meta.state == ExtensionState.LOADED

    def test_enable_disable(self, registry, brain, tmp_path):
        ext_dir = tmp_path / "exts" / "toggleable"
        ext_dir.mkdir(parents=True)

        import json
        (ext_dir / "manifest.json").write_text(json.dumps({
            "name": "toggleable",
            "module": "toolkit.py",
        }))
        (ext_dir / "toolkit.py").write_text(
            "from agno.tools import Toolkit\n"
            "class ToggleTools(Toolkit):\n"
            "    def __init__(self):\n"
            "        super().__init__(name='toggleable')\n"
        )

        registry.discover_manifests()
        assert registry.enable("toggleable")
        brain.register_toolkit.assert_called_once()

        assert registry.disable("toggleable")
        brain.unregister_toolkit.assert_called_with("ext_toggleable")

    def test_stats(self, registry):
        tk = Toolkit(name="a")
        registry.register_builtin("a", tk)
        s = registry.stats()
        assert s["total_extensions"] == 1
        assert "builtin" in s["by_source"]

    def test_record_call(self, registry):
        tk = Toolkit(name="a")
        registry.register_builtin("a", tk)
        registry.record_call("a", success=True)
        registry.record_call("a", success=False)
        meta = registry.get("a")
        assert meta.calls == 1
        assert meta.errors == 1
