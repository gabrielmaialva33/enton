import inspect
import logging
import typing
from collections.abc import Callable
from typing import Any, get_type_hints

logger = logging.getLogger(__name__)

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Callable] = {}
        self._schemas: list[dict] = []

    def register(self, func: Callable) -> Callable:
        """Decorator to register a function as a tool."""
        self._tools[func.__name__] = func
        self._schemas.append(self._generate_schema(func))
        return func

    def get_tool(self, name: str) -> Callable | None:
        return self._tools.get(name)
    
    def get_all_tools(self) -> dict[str, Callable]:
        return self._tools

    @property
    def schemas(self) -> list[dict]:
        return self._schemas

    @staticmethod
    def _parse_arg_descriptions(docstring: str) -> dict[str, str]:
        """Parse Args: section from Google-style docstrings."""
        descs: dict[str, str] = {}
        in_args = False
        for line in docstring.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("args:"):
                in_args = True
                continue
            if in_args:
                not_indented = not line.startswith(" ") and not line.startswith("\t")
                if not stripped or (not_indented and ":" not in stripped):
                    break
                if ":" in stripped:
                    name_part, _, desc_part = stripped.partition(":")
                    name_part = name_part.strip()
                    if name_part.isidentifier():
                        descs[name_part] = desc_part.strip()
        return descs

    def _generate_schema(self, func: Callable) -> dict:
        """Generates OpenAI-compatible function schema from docstring and type hints."""
        sig = inspect.signature(func)
        raw_doc = inspect.getdoc(func) or "No description provided."
        hints = get_type_hints(func)

        # Extract main description (before Args:)
        doc_lines = []
        for line in raw_doc.splitlines():
            if line.strip().lower().startswith("args:"):
                break
            doc_lines.append(line)
        description = "\n".join(doc_lines).strip() or raw_doc

        arg_descs = self._parse_arg_descriptions(raw_doc)

        parameters = {
            "type": "object",
            "properties": {},
            "required": [],
        }

        for name, param in sig.parameters.items():
            if name == "self":
                continue

            param_type = hints.get(name, Any)
            json_type = "string"

            if param_type is int:
                json_type = "integer"
            elif param_type is float:
                json_type = "number"
            elif param_type is bool:
                json_type = "boolean"
            elif param_type is list or typing.get_origin(param_type) is list:
                json_type = "array"
            elif param_type is dict or typing.get_origin(param_type) is dict:
                json_type = "object"

            parameters["properties"][name] = {
                "type": json_type,
                "description": arg_descs.get(name, f"Parameter {name}"),
            }

            if param.default == inspect.Parameter.empty:
                parameters["required"].append(name)

        return {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": description,
                "parameters": parameters,
            }
        }

# Global singleton
registry = ToolRegistry()
tool = registry.register
