"""ToolForge engine — LATM (LLM-as-Tool-Maker) code generation and validation.

Implements the closed-loop pipeline: generate → sandbox test → self-correct → deploy.
Generated skills are written to the skills directory where SkillRegistry picks
them up automatically via watchfiles.

References:
- LATM (arXiv 2305.17126) — Large Language Models as Tool Makers
- ToolMaker (arXiv 2502.11705, ACL 2025)
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import textwrap
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from enton.cognition.brain import EntonBrain

logger = logging.getLogger(__name__)

FORGE_SYSTEM_PROMPT = """\
You are a Python tool engineer for Enton, an AI robot assistant.
Generate a SINGLE Python function that will become an Agno Toolkit method.

CONSTRAINTS:
- The function must be self-contained (no external state).
- Only use Python stdlib + common packages (requests, json, re, math, os, etc).
- Do NOT use: asyncio, torch, numpy, or any heavy ML libraries.
- The function must accept string/int/float arguments and return a string.
- Write clean, safe, production-ready code.

OUTPUT FORMAT (JSON only, no markdown fences):
{
  "name": "snake_case_tool_name",
  "description": "What this tool does, in Portuguese (pt-BR)",
  "params": "param1: str, param2: int = 5",
  "code": "the function body (indented with 8 spaces, NO def line)",
  "test_code": "assert-based test code that validates the function works"
}
"""

CORRECTION_PROMPT = """\
The tool you generated failed testing. Fix the code.

ORIGINAL TASK: {task}
GENERATED CODE:
{code}

TEST CODE:
{test_code}

ERROR:
{error}

Return the SAME JSON format with corrected code. JSON only, no markdown.
"""

SKILL_TEMPLATE = '''\
"""Auto-generated skill: {name}

Created by ToolForge at {timestamp}
Description: {description}
"""
from agno.tools import Toolkit

SKILL_NAME = "{name}"
SKILL_DESCRIPTION = """{description}"""
SKILL_VERSION = "1.0"
SKILL_AUTHOR = "toolforge"


class {class_name}(Toolkit):
    """Auto-generated: {description}"""

    def __init__(self):
        super().__init__(name="{name}")
        self.register(self.{name})

    def {name}(self, {params}) -> str:
        """{description}

        Args:
            {args_doc}
        """
{code}
'''


class ForgeEngine:
    """Core LATM engine: generates, tests, and deploys tool code."""

    def __init__(
        self,
        brain: EntonBrain,
        skills_dir: Path,
        sandbox_timeout: float = 10.0,
        max_retries: int = 1,
    ) -> None:
        self._brain = brain
        self._skills_dir = Path(skills_dir).expanduser()
        self._sandbox_timeout = sandbox_timeout
        self._max_retries = max_retries
        self._stats: dict[str, dict[str, Any]] = {}

    # -- public API ---------------------------------------------------------

    async def create_tool(self, task_description: str) -> dict:
        """Full LATM pipeline: generate → test → deploy.

        Returns dict with keys: success, name, description, error.
        """
        spec = await self._generate_code(task_description)
        if spec is None:
            return {"success": False, "error": "LLM failed to generate code"}

        name = spec["name"]
        code = spec["code"]
        test_code = spec["test_code"]

        passed, output = await self._sandbox_test(
            name, spec["params"], code, test_code,
        )

        # Self-correction loop
        if not passed and self._max_retries > 0:
            logger.info("ToolForge: first attempt failed, self-correcting...")
            corrected = await self._self_correct(
                task_description, code, test_code, output,
            )
            if corrected:
                spec = corrected
                code = corrected["code"]
                test_code = corrected["test_code"]
                passed, output = await self._sandbox_test(
                    name, spec["params"], code, test_code,
                )

        if not passed:
            self._record_outcome(name, success=False)
            return {
                "success": False, "name": name,
                "error": f"Test failed: {output[:200]}",
            }

        # Deploy
        path = self._deploy(
            name=name,
            description=spec["description"],
            params=spec["params"],
            code=code,
        )
        self._record_outcome(name, success=True)
        logger.info("ToolForge deployed: %s → %s", name, path)
        return {
            "success": True,
            "name": name,
            "description": spec["description"],
        }

    def retire_tool(self, name: str) -> bool:
        """Remove a tool file from skills_dir (triggers unload via watcher)."""
        path = self._skills_dir / f"{name}.py"
        if path.exists():
            path.unlink()
            self._stats.pop(name, None)
            logger.info("ToolForge retired: %s", name)
            return True
        return False

    def get_tool_stats(self) -> list[dict]:
        """Return stats for all forged tools."""
        results = []
        for name, stat in self._stats.items():
            total = stat["success"] + stat["failure"]
            results.append({
                "name": name,
                "success_count": stat["success"],
                "failure_count": stat["failure"],
                "success_rate": stat["success"] / total if total else 1.0,
                "created_at": stat.get("created_at", 0),
            })
        return results

    # -- code generation ----------------------------------------------------

    async def _generate_code(self, task: str) -> dict | None:
        """Ask LLM to generate tool code + test."""
        prompt = f"Create a tool for the following task:\n\n{task}"
        try:
            response = await self._brain.think(
                prompt, system=FORGE_SYSTEM_PROMPT,
            )
            return self._parse_json(response)
        except Exception:
            logger.warning("ToolForge code generation failed", exc_info=True)
            return None

    async def _self_correct(
        self, task: str, code: str, test_code: str, error: str,
    ) -> dict | None:
        """Send error back to LLM for correction (1 retry)."""
        prompt = CORRECTION_PROMPT.format(
            task=task, code=code, test_code=test_code, error=error[:500],
        )
        try:
            response = await self._brain.think(
                prompt, system=FORGE_SYSTEM_PROMPT,
            )
            return self._parse_json(response)
        except Exception:
            logger.warning("ToolForge self-correction failed")
            return None

    # -- sandbox ------------------------------------------------------------

    async def _sandbox_test(
        self, name: str, params: str, code: str, test_code: str,
    ) -> tuple[bool, str]:
        """Run generated code + test in subprocess sandbox."""
        # Build standalone test script
        func_body = textwrap.indent(code, "    ")
        script = f"def {name}({params}) -> str:\n{func_body}\n\n{test_code}\nprint('PASS')\n"

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-c", script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self._sandbox_timeout,
            )
            output = (stdout or b"").decode() + (stderr or b"").decode()
            passed = proc.returncode == 0 and "PASS" in output
            return passed, output.strip()
        except TimeoutError:
            return False, "Timeout: code took too long to execute"
        except Exception as exc:
            return False, f"Sandbox error: {exc}"

    # -- deploy -------------------------------------------------------------

    def _deploy(
        self, name: str, description: str, params: str, code: str,
    ) -> Path:
        """Write validated code to skills_dir as .py file."""
        self._skills_dir.mkdir(parents=True, exist_ok=True)

        class_name = "".join(
            word.capitalize() for word in name.split("_")
        ) + "Tools"

        # Build args docstring
        args_doc = "\n            ".join(
            f"{p.strip()}: ..." for p in params.split(",") if p.strip()
        )

        # Ensure code is indented properly (8 spaces for method body)
        code_lines = code.strip().split("\n")
        indented_code = "\n".join(
            f"        {line}" for line in code_lines
        )

        content = SKILL_TEMPLATE.format(
            name=name,
            description=description,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            class_name=class_name,
            params=params,
            args_doc=args_doc or "...",
            code=indented_code,
        )

        path = self._skills_dir / f"{name}.py"
        path.write_text(content)
        return path

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        """Parse JSON from LLM response, stripping markdown fences."""
        clean = text.strip()
        if clean.startswith("```"):
            clean = re.sub(r"^```\w*\n?", "", clean)
            clean = re.sub(r"\n?```$", "", clean)

        try:
            data = json.loads(clean)
            required = {"name", "description", "code", "test_code", "params"}
            if not isinstance(data, dict) or not required.issubset(data.keys()):
                return None
            return data
        except (json.JSONDecodeError, Exception):
            return None

    def _record_outcome(self, name: str, *, success: bool) -> None:
        """Track success/failure for tool quality metrics."""
        if name not in self._stats:
            self._stats[name] = {
                "success": 0, "failure": 0, "created_at": time.time(),
            }
        if success:
            self._stats[name]["success"] += 1
        else:
            self._stats[name]["failure"] += 1
