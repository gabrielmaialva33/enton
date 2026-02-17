"""Tests for FileTools — filesystem operations toolkit."""

from __future__ import annotations

import pytest

from enton.skills._shell_state import ShellState
from enton.skills.file_toolkit import FileTools


@pytest.fixture()
def file_tools(tmp_path):
    state = ShellState(cwd=tmp_path)
    return FileTools(state)


# --- Registration ---


def test_file_tools_registers():
    ft = FileTools()
    assert ft.name == "file_tools"


# --- read_file ---


@pytest.mark.asyncio()
async def test_read_file_basic(file_tools, tmp_path):
    f = tmp_path / "hello.txt"
    f.write_text("line1\nline2\nline3\n")

    result = await file_tools.read_file("hello.txt")
    assert "3 linhas total" in result
    assert "line1" in result
    assert "line2" in result
    assert "line3" in result


@pytest.mark.asyncio()
async def test_read_file_line_range(file_tools, tmp_path):
    f = tmp_path / "range.txt"
    f.write_text("\n".join(f"L{i}" for i in range(1, 21)))

    result = await file_tools.read_file("range.txt", start_line=5, end_line=10)
    assert "mostrando 5-10" in result
    assert "L5" in result
    assert "L10" in result
    assert "L11" not in result


@pytest.mark.asyncio()
async def test_read_file_not_found(file_tools):
    result = await file_tools.read_file("nonexistent.txt")
    assert "nao existe" in result


@pytest.mark.asyncio()
async def test_read_file_blocked(file_tools):
    result = await file_tools.read_file("/etc/shadow")
    assert "BLOQUEADO" in result


@pytest.mark.asyncio()
async def test_read_file_binary(file_tools, tmp_path):
    f = tmp_path / "bin.dat"
    f.write_bytes(b"\x00\x01\x02\x03" * 100)

    result = await file_tools.read_file("bin.dat")
    assert "binario" in result.lower()


@pytest.mark.asyncio()
async def test_read_file_sensitive_warning(file_tools, tmp_path):
    d = tmp_path / ".ssh"
    d.mkdir()
    f = d / "config"
    f.write_text("Host github.com\n")

    result = await file_tools.read_file(".ssh/config")
    assert "AVISO" in result
    assert "sensivel" in result


# --- write_file ---


@pytest.mark.asyncio()
async def test_write_file_basic(file_tools, tmp_path):
    result = await file_tools.write_file("out.txt", "hello world\n")
    assert "escrito" in result.lower()

    f = tmp_path / "out.txt"
    assert f.read_text() == "hello world\n"


@pytest.mark.asyncio()
async def test_write_file_creates_dirs(file_tools, tmp_path):
    result = await file_tools.write_file("sub/dir/file.txt", "nested\n")
    assert "escrito" in result.lower()

    f = tmp_path / "sub" / "dir" / "file.txt"
    assert f.exists()


@pytest.mark.asyncio()
async def test_write_file_blocked(file_tools):
    result = await file_tools.write_file("/etc/test.conf", "nope")
    assert "BLOQUEADO" in result


# --- edit_file ---


@pytest.mark.asyncio()
async def test_edit_file_basic(file_tools, tmp_path):
    f = tmp_path / "edit.py"
    f.write_text("x = 1\ny = 2\nz = 3\n")

    result = await file_tools.edit_file("edit.py", "y = 2", "y = 42")
    assert "Editado" in result

    assert "y = 42" in f.read_text()
    assert "y = 2" not in f.read_text()


@pytest.mark.asyncio()
async def test_edit_file_not_found_text(file_tools, tmp_path):
    f = tmp_path / "edit2.py"
    f.write_text("foo = 1\nbar = 2\n")

    result = await file_tools.edit_file("edit2.py", "baz = 999", "qux = 0")
    assert "nao encontrado" in result.lower()


@pytest.mark.asyncio()
async def test_edit_file_multiple_occurrences(file_tools, tmp_path):
    f = tmp_path / "multi.txt"
    f.write_text("abc\nabc\nabc\n")

    result = await file_tools.edit_file("multi.txt", "abc", "XYZ")
    assert "1a de 3" in result

    content = f.read_text()
    assert content.count("XYZ") == 1
    assert content.count("abc") == 2


@pytest.mark.asyncio()
async def test_edit_file_blocked(file_tools):
    result = await file_tools.edit_file("/usr/bin/test", "a", "b")
    assert "BLOQUEADO" in result


# --- find_files ---


@pytest.mark.asyncio()
async def test_find_files(file_tools, tmp_path):
    (tmp_path / "a.py").write_text("pass")
    (tmp_path / "b.py").write_text("pass")
    (tmp_path / "c.txt").write_text("nope")

    result = await file_tools.find_files("*.py")
    assert "2 resultado" in result
    assert "a.py" in result
    assert "b.py" in result
    assert "c.txt" not in result


@pytest.mark.asyncio()
async def test_find_files_recursive(file_tools, tmp_path):
    sub = tmp_path / "pkg"
    sub.mkdir()
    (sub / "mod.py").write_text("pass")

    result = await file_tools.find_files("**/*.py")
    assert "mod.py" in result


@pytest.mark.asyncio()
async def test_find_files_no_results(file_tools):
    result = await file_tools.find_files("*.nonexistent")
    assert "Nenhum" in result


# --- search_in_files ---


@pytest.mark.asyncio()
async def test_search_in_files(file_tools, tmp_path):
    (tmp_path / "a.py").write_text("import os\nimport sys\n")
    (tmp_path / "b.py").write_text("x = 1\n")

    result = await file_tools.search_in_files("import", file_glob="*.py")
    assert "a.py" in result
    assert "import os" in result
    assert "b.py" not in result


@pytest.mark.asyncio()
async def test_search_in_files_regex(file_tools, tmp_path):
    (tmp_path / "code.py").write_text("def foo():\n    pass\ndef bar():\n    pass\n")

    result = await file_tools.search_in_files(r"def \w+\(\)", file_glob="*.py")
    assert "def foo()" in result
    assert "def bar()" in result


@pytest.mark.asyncio()
async def test_search_in_files_no_match(file_tools, tmp_path):
    (tmp_path / "empty.py").write_text("pass\n")

    result = await file_tools.search_in_files("nonexistent_pattern", file_glob="*.py")
    assert "Nenhum" in result


# --- list_directory ---


@pytest.mark.asyncio()
async def test_list_directory(file_tools, tmp_path):
    (tmp_path / "file.txt").write_text("data")
    (tmp_path / "subdir").mkdir()

    result = await file_tools.list_directory()
    assert "file.txt" in result
    assert "subdir" in result
    assert "[dir]" in result
    assert "[file]" in result


@pytest.mark.asyncio()
async def test_list_directory_empty(file_tools, tmp_path):
    empty = tmp_path / "empty_dir"
    empty.mkdir()

    result = await file_tools.list_directory("empty_dir")
    assert "vazio" in result
