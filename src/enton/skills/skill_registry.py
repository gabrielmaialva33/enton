"""SkillRegistry — file-system watcher for hot-loading dynamic Agno Toolkits.

Watches ``~/.enton/skills/`` (configurable) for ``.py`` files and loads them
as Agno Toolkits on-the-fly.  A valid skill module must either:

1. Define a module-level ``create_toolkit()`` function returning a ``Toolkit``, or
2. Contain a class that is a direct subclass of ``Toolkit``.

Uses ``watchfiles.awatch`` (Rust-backed inotify) for near-zero overhead.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from agno.tools import Toolkit

from enton.core.events import SkillEvent
from enton.skills.skill_protocol import SkillMetadata

if TYPE_CHECKING:
    import types

    from enton.cognition.brain import EntonBrain
    from enton.core.events import EventBus

logger = logging.getLogger(__name__)


class SkillRegistry:
    """Watches a directory for .py skill files and hot-loads them."""

    def __init__(
        self,
        brain: EntonBrain,
        bus: EventBus,
        skills_dir: Path | str = "~/.enton/skills",
    ) -> None:
        self._brain = brain
        self._bus = bus
        self._skills_dir = Path(skills_dir).expanduser()
        self._loaded: dict[str, SkillMetadata] = {}
        self._toolkits: dict[str, Toolkit] = {}

    # -- public API ---------------------------------------------------------

    @property
    def loaded_skills(self) -> dict[str, SkillMetadata]:
        return dict(self._loaded)

    def list_skills(self) -> list[str]:
        return list(self._loaded.keys())

    # -- lifecycle ----------------------------------------------------------

    async def load_skill(self, path: Path) -> bool:
        """Load a single .py file as a dynamic skill."""
        name = path.stem
        module = self._import_module(path)
        if module is None:
            return False

        toolkit = self._extract_toolkit(module)
        if toolkit is None:
            logger.warning("No Toolkit found in %s", path.name)
            self._cleanup_module(name)
            return False

        # If already loaded, unload first
        if name in self._toolkits:
            await self.unload_skill(name)

        self._toolkits[name] = toolkit
        self._loaded[name] = SkillMetadata(
            name=name,
            file_path=str(path),
            description=getattr(module, "SKILL_DESCRIPTION", ""),
            author=getattr(module, "SKILL_AUTHOR", "unknown"),
            version=getattr(module, "SKILL_VERSION", "1.0"),
        )
        self._brain.register_toolkit(toolkit, name)
        await self._bus.emit(SkillEvent(kind="loaded", name=name))
        logger.info("Loaded dynamic skill: %s from %s", name, path.name)
        return True

    async def unload_skill(self, name: str) -> bool:
        """Unload a skill by name."""
        toolkit = self._toolkits.pop(name, None)
        if toolkit is None:
            return False
        self._brain.unregister_toolkit(name)
        self._loaded.pop(name, None)
        self._cleanup_module(name)
        await self._bus.emit(SkillEvent(kind="unloaded", name=name))
        logger.info("Unloaded dynamic skill: %s", name)
        return True

    async def reload_skill(self, path: Path) -> bool:
        """Unload then reload a modified skill file."""
        name = path.stem
        await self.unload_skill(name)
        return await self.load_skill(path)

    # -- background loop (TaskGroup) ----------------------------------------

    async def run(self) -> None:
        """Main loop: watch skills_dir with watchfiles.awatch()."""
        self._skills_dir.mkdir(parents=True, exist_ok=True)
        await self._scan_existing()

        try:
            from watchfiles import Change, awatch
        except ImportError:
            logger.warning("watchfiles not installed — skill hot-reload disabled")
            return

        logger.info("SkillRegistry watching %s", self._skills_dir)
        async for changes in awatch(self._skills_dir):
            for change_type, path_str in changes:
                path = Path(path_str)
                if path.suffix != ".py" or path.name.startswith("_"):
                    continue
                try:
                    if change_type == Change.added:
                        await self.load_skill(path)
                    elif change_type == Change.modified:
                        await self.reload_skill(path)
                    elif change_type == Change.deleted:
                        await self.unload_skill(path.stem)
                except Exception:
                    logger.exception("SkillRegistry error for %s", path.name)

    # -- internals ----------------------------------------------------------

    def _import_module(self, path: Path) -> types.ModuleType | None:
        """Import a .py file as a module (no bytecode cache)."""
        import types as _types

        module_name = f"enton_skill_{path.stem}"
        sys.modules.pop(module_name, None)
        try:
            source = path.read_text()
            code = compile(source, str(path), "exec")
            module = _types.ModuleType(module_name)
            module.__file__ = str(path)
            sys.modules[module_name] = module
            exec(code, module.__dict__)  # noqa: S102
            return module
        except Exception:
            logger.warning("Failed to import %s", path.name, exc_info=True)
            self._cleanup_module(path.stem)
            return None

    def _extract_toolkit(self, module: types.ModuleType) -> Toolkit | None:
        """Extract Toolkit from module via create_toolkit() or class scan."""
        # Method 1: module-level factory function
        factory = getattr(module, "create_toolkit", None)
        if callable(factory):
            try:
                result = factory()
                if isinstance(result, Toolkit):
                    return result
            except Exception:
                logger.warning("create_toolkit() failed in %s", module.__name__)

        # Method 2: scan for Toolkit subclass
        for attr_name in dir(module):
            attr = getattr(module, attr_name, None)
            if (
                isinstance(attr, type)
                and issubclass(attr, Toolkit)
                and attr is not Toolkit
            ):
                try:
                    return attr()
                except Exception:
                    logger.warning("Failed to instantiate %s", attr_name)

        return None

    async def _scan_existing(self) -> None:
        """Load all .py files already present in skills_dir."""
        if not self._skills_dir.exists():
            return
        for path in sorted(self._skills_dir.glob("*.py")):
            if not path.name.startswith("_"):
                await self.load_skill(path)

    @staticmethod
    def _cleanup_module(name: str) -> None:
        """Remove dynamically loaded module from sys.modules."""
        key = f"enton_skill_{name}"
        sys.modules.pop(key, None)

    def record_outcome(self, name: str, success: bool) -> None:
        """Track success/failure for a dynamic skill."""
        meta = self._loaded.get(name)
        if meta is None:
            return
        if success:
            meta.success_count += 1
        else:
            meta.failure_count += 1
