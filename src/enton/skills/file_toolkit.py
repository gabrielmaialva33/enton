"""Agno Toolkit for filesystem operations — read, write, edit, find, grep."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from agno.tools import Toolkit

from enton.skills._shell_state import ShellState

logger = logging.getLogger(__name__)

_MAX_READ_LINES = 2000
_MAX_LINE_LEN = 500
_MAX_OUTPUT = 8000
_MAX_SEARCH_RESULTS = 50

# Paths blocked from reading
_READ_BLOCKED = frozenset(
    {
        "/etc/shadow",
        "/etc/gshadow",
        "/proc/kcore",
        "/dev/mem",
        "/dev/kmem",
    }
)

# Path prefixes blocked from writing
_WRITE_BLOCKED_PREFIXES = (
    "/etc/",
    "/usr/",
    "/bin/",
    "/sbin/",
    "/boot/",
    "/proc/",
    "/sys/",
)

# Paths that generate warnings (sensitive but not blocked)
_SENSITIVE_PATTERNS = (".ssh/", ".gnupg/", ".aws/", ".env", "credentials.json")


def _is_binary(path: Path, sample_size: int = 8192) -> bool:
    """Check if file is binary by looking for null bytes."""
    try:
        with path.open("rb") as f:
            chunk = f.read(sample_size)
        return b"\x00" in chunk
    except OSError:
        return True


def _check_sensitive(path: Path) -> str | None:
    """Return warning string if path is sensitive."""
    s = str(path)
    for pat in _SENSITIVE_PATTERNS:
        if pat in s:
            return f"AVISO: Arquivo sensivel detectado ({pat})"
    return None


class FileTools(Toolkit):
    """Filesystem operations: read, write, edit, find, grep."""

    def __init__(self, state: ShellState | None = None):
        super().__init__(name="file_tools")
        self._state = state or ShellState()
        self.register(self.read_file)
        self.register(self.write_file)
        self.register(self.edit_file)
        self.register(self.find_files)
        self.register(self.search_in_files)
        self.register(self.list_directory)

    def _resolve(self, path: str) -> Path:
        return self._state.resolve_path(path)

    # ------------------------------------------------------------------
    # read_file
    # ------------------------------------------------------------------

    async def read_file(self, path: str, start_line: int = 0, end_line: int = 0) -> str:
        """Le o conteudo de um arquivo com numeros de linha.

        Args:
            path: Caminho do arquivo (absoluto ou relativo ao cwd).
            start_line: Linha inicial (1-based, 0 = inicio).
            end_line: Linha final (1-based, 0 = ate o fim, max 2000 linhas).
        """
        p = self._resolve(path)

        if str(p) in _READ_BLOCKED:
            return f"BLOQUEADO: Leitura de '{p}' nao permitida."

        if not p.exists():
            return f"ERRO: Arquivo '{p}' nao existe."

        if not p.is_file():
            return f"ERRO: '{p}' nao e um arquivo."

        if _is_binary(p):
            size = p.stat().st_size
            return f"Arquivo binario: {p} ({size:,} bytes)"

        warning = _check_sensitive(p)

        try:
            lines = p.read_text(errors="replace").splitlines()
        except OSError as e:
            return f"ERRO: {e}"

        total = len(lines)
        start = max(0, start_line - 1) if start_line > 0 else 0
        end = min(total, end_line) if end_line > 0 else total
        end = min(end, start + _MAX_READ_LINES)

        numbered = []
        for i, line in enumerate(lines[start:end], start=start + 1):
            if len(line) > _MAX_LINE_LEN:
                line = line[:_MAX_LINE_LEN] + "..."
            numbered.append(f"{i:>5}\t{line}")

        result = "\n".join(numbered)
        if len(result) > _MAX_OUTPUT:
            result = result[:_MAX_OUTPUT] + "\n... (truncado)"

        header = f"[{p}] ({total} linhas total)"
        if start_line or end_line:
            header += f" mostrando {start + 1}-{end}"

        parts = [header]
        if warning:
            parts.append(warning)
        parts.append(result)
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # write_file
    # ------------------------------------------------------------------

    async def write_file(self, path: str, content: str) -> str:
        """Cria ou sobrescreve um arquivo com o conteudo fornecido.

        Args:
            path: Caminho do arquivo (absoluto ou relativo ao cwd).
            content: Conteudo completo do arquivo.
        """
        p = self._resolve(path)

        for prefix in _WRITE_BLOCKED_PREFIXES:
            if str(p).startswith(prefix):
                return f"BLOQUEADO: Escrita em '{prefix}' nao permitida."

        warning = _check_sensitive(p)

        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
        except OSError as e:
            return f"ERRO: {e}"

        lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        result = f"Arquivo escrito: {p} ({lines} linhas, {len(content)} bytes)"
        if warning:
            result = f"{warning}\n{result}"
        return result

    # ------------------------------------------------------------------
    # edit_file
    # ------------------------------------------------------------------

    async def edit_file(self, path: str, old_text: str, new_text: str) -> str:
        """Edita um arquivo substituindo texto exato (search-replace).

        Substitui a PRIMEIRA ocorrencia de old_text por new_text.
        O old_text deve ser uma copia exata do trecho a substituir.

        Args:
            path: Caminho do arquivo.
            old_text: Texto exato a ser encontrado e substituido.
            new_text: Texto substituto.
        """
        p = self._resolve(path)

        for prefix in _WRITE_BLOCKED_PREFIXES:
            if str(p).startswith(prefix):
                return f"BLOQUEADO: Escrita em '{prefix}' nao permitida."

        if not p.exists():
            return f"ERRO: Arquivo '{p}' nao existe."

        try:
            content = p.read_text(errors="replace")
        except OSError as e:
            return f"ERRO: {e}"

        count = content.count(old_text)
        if count == 0:
            return self._edit_diagnostic(content, old_text, p)

        new_content = content.replace(old_text, new_text, 1)

        try:
            p.write_text(new_content)
        except OSError as e:
            return f"ERRO ao salvar: {e}"

        msg = f"Editado: {p}"
        if count > 1:
            msg += f" (substituida 1a de {count} ocorrencias)"
        return msg

    @staticmethod
    def _edit_diagnostic(content: str, old_text: str, path: Path) -> str:
        """Help diagnose why old_text wasn't found."""
        lines = content.splitlines()
        # Normalize whitespace for fuzzy search
        old_norm = " ".join(old_text.split())

        best_line = 0
        best_ratio = 0.0
        for i, line in enumerate(lines, 1):
            line_norm = " ".join(line.split())
            if not line_norm or not old_norm:
                continue
            # Simple overlap ratio
            common = sum(1 for c in old_norm if c in line_norm)
            ratio = common / max(len(old_norm), len(line_norm))
            if ratio > best_ratio:
                best_ratio = ratio
                best_line = i

        msg = f"ERRO: Texto nao encontrado em '{path}'."
        if best_ratio > 0.5:
            ctx = lines[best_line - 1] if best_line > 0 else ""
            msg += f"\nLinha {best_line} tem algo parecido: {ctx[:200]}"
        msg += "\nDica: o old_text deve ser copia EXATA (incluindo espacos e quebras)."
        return msg

    # ------------------------------------------------------------------
    # find_files
    # ------------------------------------------------------------------

    async def find_files(self, pattern: str, directory: str = "") -> str:
        """Busca arquivos por glob pattern (ex: '**/*.py', '*.json').

        Args:
            pattern: Glob pattern para buscar.
            directory: Diretorio base (default: cwd).
        """
        base = self._resolve(directory) if directory else self._state.cwd

        if not base.is_dir():
            return f"ERRO: Diretorio '{base}' nao existe."

        results = []
        try:
            for p in sorted(base.glob(pattern)):
                if len(results) >= _MAX_SEARCH_RESULTS:
                    results.append(f"... (limitado a {_MAX_SEARCH_RESULTS} resultados)")
                    break
                rel = p.relative_to(base) if p.is_relative_to(base) else p
                size = p.stat().st_size if p.is_file() else 0
                kind = "d" if p.is_dir() else "f"
                if kind == "f":
                    results.append(f"[f] {rel}  ({size:,} bytes)")
                else:
                    results.append(f"[d] {rel}/")
        except OSError as e:
            return f"ERRO: {e}"

        if not results:
            return f"Nenhum arquivo encontrado para '{pattern}' em {base}"

        return f"[{base}] {len(results)} resultado(s):\n" + "\n".join(results)

    # ------------------------------------------------------------------
    # search_in_files
    # ------------------------------------------------------------------

    async def search_in_files(
        self, pattern: str, directory: str = "", file_glob: str = "**/*"
    ) -> str:
        """Busca texto/regex dentro de arquivos (grep).

        Args:
            pattern: Regex para buscar no conteudo dos arquivos.
            directory: Diretorio base (default: cwd).
            file_glob: Glob para filtrar quais arquivos (default: todos).
        """
        base = self._resolve(directory) if directory else self._state.cwd

        if not base.is_dir():
            return f"ERRO: Diretorio '{base}' nao existe."

        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return f"ERRO regex: {e}"

        results = []
        files_checked = 0

        for p in sorted(base.glob(file_glob)):
            if not p.is_file() or _is_binary(p):
                continue

            files_checked += 1
            try:
                text = p.read_text(errors="replace")
            except OSError:
                continue

            for i, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    rel = p.relative_to(base) if p.is_relative_to(base) else p
                    preview = line.strip()[:120]
                    results.append(f"{rel}:{i}  {preview}")
                    if len(results) >= _MAX_SEARCH_RESULTS:
                        break

            if len(results) >= _MAX_SEARCH_RESULTS:
                results.append(f"... (limitado a {_MAX_SEARCH_RESULTS} resultados)")
                break

        if not results:
            return f"Nenhum match para '{pattern}' em {files_checked} arquivo(s)"

        output = "\n".join(results)
        if len(output) > _MAX_OUTPUT:
            output = output[:_MAX_OUTPUT] + "\n... (truncado)"
        return f"[{base}] {len(results)} match(es):\n{output}"

    # ------------------------------------------------------------------
    # list_directory
    # ------------------------------------------------------------------

    async def list_directory(self, path: str = "") -> str:
        """Lista arquivos e pastas de um diretorio.

        Args:
            path: Diretorio para listar (default: cwd).
        """
        base = self._resolve(path) if path else self._state.cwd

        if not base.is_dir():
            return f"ERRO: '{base}' nao e um diretorio."

        entries = []
        try:
            for entry in sorted(base.iterdir()):
                try:
                    if entry.is_dir():
                        entries.append(f"  [dir]  {entry.name}/")
                    elif entry.is_symlink():
                        target = os.readlink(entry)
                        entries.append(f"  [lnk]  {entry.name} -> {target}")
                    else:
                        size = entry.stat().st_size
                        entries.append(f"  [file] {entry.name}  ({size:,} bytes)")
                except OSError:
                    entries.append(f"  [???]  {entry.name}")
        except OSError as e:
            return f"ERRO: {e}"

        if not entries:
            return f"{base}/ (vazio)"

        return f"{base}/ ({len(entries)} itens):\n" + "\n".join(entries)
