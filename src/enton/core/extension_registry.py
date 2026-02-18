"""Extension Registry — centralized plugin system for Enton.

Extends the existing SkillRegistry (filesystem watcher) with:
- Entry point loading (pip packages with 'enton.extensions' group)
- Manifest-based extensions (JSON metadata + toolkit module)
- Remote extension install (git clone → load)
- Dependency tracking and conflict detection
- Centralized enable/disable/list API

Inspired by Goose's MCP-first plugin design and VSCode's extension model.

Architecture:
    ExtensionRegistry (this) ←→ SkillRegistry (filesystem watcher)
                              ←→ EntonBrain (register_toolkit)

    ExtensionRegistry manages the "what" (discovery, metadata, lifecycle).
    SkillRegistry manages the "how" (import, instantiate, hot-reload).
    Brain manages the "where" (Agent tool list).
"""

from __future__ import annotations

import importlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agno.tools import Toolkit

if TYPE_CHECKING:
    from enton.cognition.brain import EntonBrain

logger = logging.getLogger(__name__)


class ExtensionSource(StrEnum):
    """Where an extension came from."""

    BUILTIN = "builtin"  # shipped with Enton
    FILESYSTEM = "filesystem"  # loaded from skills_dir (SkillRegistry)
    ENTRYPOINT = "entrypoint"  # pip package entry point
    MANIFEST = "manifest"  # local manifest JSON
    REMOTE = "remote"  # git cloned


class ExtensionState(StrEnum):
    """Extension lifecycle state."""

    DISCOVERED = "discovered"
    LOADED = "loaded"
    ENABLED = "enabled"
    DISABLED = "disabled"
    ERROR = "error"


@dataclass
class ExtensionMeta:
    """Rich metadata for an extension."""

    name: str
    source: ExtensionSource
    state: ExtensionState = ExtensionState.DISCOVERED

    # Metadata
    description: str = ""
    version: str = "0.1.0"
    author: str = "unknown"
    tags: list[str] = field(default_factory=list)

    # Runtime
    toolkit: Toolkit | None = field(default=None, repr=False)
    module_path: str = ""
    error: str = ""
    loaded_at: float = 0.0
    tool_count: int = 0

    # Stats
    calls: int = 0
    errors: int = 0

    @property
    def success_rate(self) -> float:
        total = self.calls + self.errors
        return self.calls / total if total else 1.0

    def summary(self) -> str:
        status = f"[{self.state}]"
        tools = f"{self.tool_count} tools" if self.tool_count else "no tools"
        return f"{self.name} {status} ({self.source}, {tools}, v{self.version})"


# Manifest schema (JSON file alongside toolkit .py)
MANIFEST_EXAMPLE = {
    "name": "my_extension",
    "version": "1.0.0",
    "description": "What it does",
    "author": "someone",
    "tags": ["utility"],
    "module": "my_toolkit.py",  # relative to manifest dir
    "dependencies": [],  # pip packages
}


class ExtensionRegistry:
    """Centralized extension management for Enton.

    Discovers, loads, and manages lifecycle of toolkit extensions.
    Works alongside SkillRegistry (which handles filesystem watching).
    """

    def __init__(
        self,
        brain: EntonBrain,
        extensions_dir: Path | str = "~/.enton/extensions",
    ) -> None:
        self._brain = brain
        self._extensions_dir = Path(extensions_dir).expanduser()
        self._extensions: dict[str, ExtensionMeta] = {}

    # ------------------------------------------------------------------ #
    # Discovery
    # ------------------------------------------------------------------ #

    def discover_entrypoints(self) -> list[str]:
        """Discover extensions registered as pip entry points.

        Extensions declare: [project.entry-points."enton.extensions"]
        my_ext = "my_package.toolkit:create_toolkit"
        """
        discovered: list[str] = []
        try:
            if hasattr(importlib.metadata, "entry_points"):
                eps = importlib.metadata.entry_points()
                # Python 3.12+: eps is a dict-like
                group = eps.get("enton.extensions", [])
                # Python 3.9-3.11 compat
                if hasattr(eps, "select"):
                    group = eps.select(group="enton.extensions")

                for ep in group:
                    name = ep.name
                    if name not in self._extensions:
                        self._extensions[name] = ExtensionMeta(
                            name=name,
                            source=ExtensionSource.ENTRYPOINT,
                            module_path=str(ep.value),
                        )
                        discovered.append(name)
                        logger.info("Discovered entry point extension: %s", name)
        except Exception:
            logger.debug("Entry point discovery failed", exc_info=True)

        return discovered

    def discover_manifests(self) -> list[str]:
        """Discover extensions from manifest.json files in extensions_dir."""
        discovered: list[str] = []
        self._extensions_dir.mkdir(parents=True, exist_ok=True)

        for manifest_path in self._extensions_dir.glob("*/manifest.json"):
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
                name = data.get("name", manifest_path.parent.name)
                if name in self._extensions:
                    continue

                module_file = data.get("module", "toolkit.py")
                module_path = str(manifest_path.parent / module_file)

                self._extensions[name] = ExtensionMeta(
                    name=name,
                    source=ExtensionSource.MANIFEST,
                    description=data.get("description", ""),
                    version=data.get("version", "0.1.0"),
                    author=data.get("author", "unknown"),
                    tags=data.get("tags", []),
                    module_path=module_path,
                )
                discovered.append(name)
                logger.info("Discovered manifest extension: %s", name)
            except (json.JSONDecodeError, KeyError):
                logger.warning("Invalid manifest: %s", manifest_path)

        return discovered

    def discover_all(self) -> list[str]:
        """Run all discovery mechanisms."""
        found: list[str] = []
        found.extend(self.discover_entrypoints())
        found.extend(self.discover_manifests())
        return found

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    def load(self, name: str) -> bool:
        """Load a discovered extension into memory."""
        meta = self._extensions.get(name)
        if not meta:
            logger.warning("Extension '%s' not found", name)
            return False

        if meta.state == ExtensionState.ENABLED:
            return True  # already loaded

        try:
            toolkit = self._load_toolkit(meta)
            if toolkit is None:
                meta.state = ExtensionState.ERROR
                meta.error = "Failed to extract toolkit from module"
                return False

            meta.toolkit = toolkit
            meta.loaded_at = time.time()
            meta.tool_count = len(toolkit.functions) if hasattr(toolkit, "functions") else 0
            meta.state = ExtensionState.LOADED
            logger.info("Loaded extension: %s (%d tools)", name, meta.tool_count)
            return True

        except Exception as exc:
            meta.state = ExtensionState.ERROR
            meta.error = str(exc)[:200]
            logger.warning("Failed to load extension '%s': %s", name, exc)
            return False

    def _load_toolkit(self, meta: ExtensionMeta) -> Toolkit | None:
        """Load toolkit from module path or entry point."""
        if meta.source == ExtensionSource.ENTRYPOINT:
            return self._load_from_entrypoint(meta.module_path)

        if meta.module_path and Path(meta.module_path).exists():
            return self._load_from_file(Path(meta.module_path))

        return None

    def _load_from_entrypoint(self, ep_value: str) -> Toolkit | None:
        """Load from entry point string like 'package.module:factory'."""
        try:
            module_path, _, attr_name = ep_value.partition(":")
            module = importlib.import_module(module_path)
            factory = getattr(module, attr_name or "create_toolkit")
            result = factory()
            if isinstance(result, Toolkit):
                return result
        except Exception:
            logger.warning("Entry point load failed: %s", ep_value)
        return None

    def _load_from_file(self, path: Path) -> Toolkit | None:
        """Load toolkit from a .py file (same logic as SkillRegistry)."""
        import types

        module_name = f"enton_ext_{path.stem}"
        try:
            source = path.read_text()
            code = compile(source, str(path), "exec")
            module = types.ModuleType(module_name)
            module.__file__ = str(path)
            exec(code, module.__dict__)

            # Try factory function first
            factory = getattr(module, "create_toolkit", None)
            if callable(factory):
                result = factory()
                if isinstance(result, Toolkit):
                    return result

            # Scan for Toolkit subclass
            for attr_name in dir(module):
                attr = getattr(module, attr_name, None)
                if isinstance(attr, type) and issubclass(attr, Toolkit) and attr is not Toolkit:
                    return attr()

        except Exception:
            logger.warning("File load failed: %s", path)

        return None

    # ------------------------------------------------------------------ #
    # Enable / Disable
    # ------------------------------------------------------------------ #

    def enable(self, name: str) -> bool:
        """Enable an extension: load + register with brain."""
        meta = self._extensions.get(name)
        if not meta:
            return False

        if meta.state != ExtensionState.LOADED:
            if not self.load(name):
                return False
            meta = self._extensions[name]

        if meta.toolkit is None:
            return False

        self._brain.register_toolkit(meta.toolkit, f"ext_{name}")
        meta.state = ExtensionState.ENABLED
        logger.info("Enabled extension: %s", name)
        return True

    def disable(self, name: str) -> bool:
        """Disable an extension: unregister from brain."""
        meta = self._extensions.get(name)
        if not meta or meta.state != ExtensionState.ENABLED:
            return False

        self._brain.unregister_toolkit(f"ext_{name}")
        meta.state = ExtensionState.DISABLED
        meta.toolkit = None
        logger.info("Disabled extension: %s", name)
        return True

    # ------------------------------------------------------------------ #
    # Install (remote)
    # ------------------------------------------------------------------ #

    async def install_from_git(self, repo_url: str, name: str = "") -> bool:
        """Clone a git repo into extensions_dir and discover it.

        Expects repo to contain manifest.json at root.
        """
        import asyncio

        if not name:
            # Extract name from URL: https://github.com/user/enton-ext-foo → foo
            name = repo_url.rstrip("/").split("/")[-1]
            name = name.removeprefix("enton-ext-").removeprefix("enton-")

        target = self._extensions_dir / name
        if target.exists():
            logger.warning("Extension dir already exists: %s", target)
            return False

        try:
            proc = await asyncio.create_subprocess_exec(
                "git",
                "clone",
                "--depth=1",
                repo_url,
                str(target),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
            if proc.returncode != 0:
                logger.warning("Git clone failed: %s", stderr.decode()[:200])
                return False

            # Discover the newly cloned extension
            self.discover_manifests()
            if name in self._extensions:
                return self.enable(name)

            logger.warning("No manifest.json found in cloned repo: %s", name)
            return False

        except Exception:
            logger.warning("Git install failed for %s", repo_url, exc_info=True)
            return False

    # ------------------------------------------------------------------ #
    # Query
    # ------------------------------------------------------------------ #

    def get(self, name: str) -> ExtensionMeta | None:
        return self._extensions.get(name)

    def list_extensions(
        self,
        state: ExtensionState | None = None,
        source: ExtensionSource | None = None,
    ) -> list[ExtensionMeta]:
        """List extensions with optional filters."""
        exts = list(self._extensions.values())
        if state:
            exts = [e for e in exts if e.state == state]
        if source:
            exts = [e for e in exts if e.source == source]
        return sorted(exts, key=lambda e: e.name)

    def register_builtin(self, name: str, toolkit: Toolkit) -> None:
        """Register a builtin toolkit for tracking (not loading)."""
        tool_count = len(toolkit.functions) if hasattr(toolkit, "functions") else 0
        self._extensions[name] = ExtensionMeta(
            name=name,
            source=ExtensionSource.BUILTIN,
            state=ExtensionState.ENABLED,
            toolkit=toolkit,
            loaded_at=time.time(),
            tool_count=tool_count,
        )

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #

    def record_call(self, ext_name: str, success: bool = True) -> None:
        """Record a tool call for an extension."""
        meta = self._extensions.get(ext_name)
        if not meta:
            return
        if success:
            meta.calls += 1
        else:
            meta.errors += 1

    def stats(self) -> dict[str, Any]:
        """Registry-wide statistics."""
        by_state: dict[str, int] = {}
        by_source: dict[str, int] = {}
        total_tools = 0

        for ext in self._extensions.values():
            by_state[ext.state] = by_state.get(ext.state, 0) + 1
            by_source[ext.source] = by_source.get(ext.source, 0) + 1
            if ext.state == ExtensionState.ENABLED:
                total_tools += ext.tool_count

        return {
            "total_extensions": len(self._extensions),
            "total_tools": total_tools,
            "by_state": by_state,
            "by_source": by_source,
        }

    def summary(self) -> str:
        s = self.stats()
        enabled = s["by_state"].get("enabled", 0)
        return (
            f"Extensions: {s['total_extensions']} total, "
            f"{enabled} enabled, {s['total_tools']} tools"
        )
