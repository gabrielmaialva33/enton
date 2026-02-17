"""Advanced multi-language coding toolkit — gives Enton coding superpowers.

Enton can compile, run, analyze, and optimize code in C, Rust, Zig,
Python, Erlang, Elixir, and more. Includes a built-in knowledge base
of advanced patterns for each language.

Inspired by OpenClaw skills + polyglot systems programming expertise.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
import time
from pathlib import Path

from agno.tools import Toolkit

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language registry — compilers, flags, file extensions
# ---------------------------------------------------------------------------

_LANGS: dict[str, dict[str, str | list[str]]] = {
    "c": {
        "ext": ".c",
        "compile": "gcc -O2 -Wall -Wextra -std=c23 -o {out} {src} -lm -lpthread",
        "run": "{out}",
        "name": "C (GCC 15)",
    },
    "rust": {
        "ext": ".rs",
        "compile": "rustc -O -o {out} {src}",
        "run": "{out}",
        "name": "Rust 1.93",
    },
    "zig": {
        "ext": ".zig",
        "compile": "",  # zig run compiles+runs in one step
        "run": "zig run {src}",
        "name": "Zig 0.15",
    },
    "python": {
        "ext": ".py",
        "compile": "",
        "run": "python3 {src}",
        "name": "Python 3.14",
    },
    "erlang": {
        "ext": ".erl",
        "compile": "erlc -o {dir} {src}",
        "run": "erl -pa {dir} -noshell -eval '{module}:main(), halt().'",
        "name": "Erlang/OTP 28",
    },
    "elixir": {
        "ext": ".exs",
        "compile": "",
        "run": "elixir {src}",
        "name": "Elixir 1.19",
    },
}

# ---------------------------------------------------------------------------
# Built-in knowledge base — advanced patterns per language
# ---------------------------------------------------------------------------

_KNOWLEDGE: dict[str, str] = {
    "c": """\
== C Advanced Patterns ==
• Lock-free MPMC ring buffer: atomic CAS on head/tail, cache-line aligned slots
• Epoch-based memory reclamation: safe deferred free without GC
• Intrusive linked lists: container_of() macro, zero-alloc node embedding
• Coroutines via setjmp/longjmp or __attribute__((cleanup))
• Object-oriented C: vtable structs, opaque handles, _Generic dispatch (C11)
• Zero-copy I/O: splice/sendfile/io_uring for kernel-bypass networking
• Ring buffer ISR↔main: single-producer/single-consumer, volatile + memory barriers
• Linker scripts: custom .text/.data/.bss sections, overlay memory maps
• Fixed-point arithmetic: Q16.16 format, saturation ops for DSP/embedded
• Compile-time asserts: _Static_assert, BUILD_BUG_ON_ZERO pattern
• Flexible array members (FAM): struct trailing array, cache-friendly layout
• Signal-safe programming: sig_atomic_t, self-pipe trick, signalfd""",
    "rust": """\
== Rust Advanced Patterns ==
• Zero-cost abstractions: iterators compile to same ASM as manual loops
• Trait specialization (nightly): default impl + specialized impl per type
• Pin<Box<T>>: self-referential structs for async state machines
• Tower middleware: Service trait, Layer composition for networking
• Typestate pattern: encode state machine in types (compile-time FSM)
• GATs (Generic Associated Types): async trait returns, lending iterators
• const generics: array sizes as type params, compile-time matrix dims
• #[repr(C)] + FFI: safe C interop, cbindgen for header generation
• Async traits (AFIT): native async fn in traits (stable since 1.75)
• Rayon par_iter: data parallelism, work-stealing thread pool
• Pin + Unpin: manual Future::poll state machines for zero-alloc async
• Error handling: thiserror for libs, anyhow for apps, ? operator chains
• Serde: zero-copy deserialization with #[serde(borrow)]
• Tokio: select!, spawn, JoinSet for structured concurrency""",
    "zig": """\
== Zig Advanced Patterns ==
• comptime: move computation to compile-time (40% runtime savings vs C)
• comptime ORM: generate DB schema types at compile time
• Allocator interface: std.mem.Allocator is a trait object (vtable)
• No hidden control flow: no exceptions, no hidden allocations
• @import + build.zig: hermetic, reproducible builds, custom codegen
• comptime string formatting: type-safe printf at zero runtime cost
• Error unions: error{OutOfMemory}!T replaces C errno + NULL checks
• Packed structs: bit-level layout control for protocols/hardware
• @cImport: seamless C header inclusion, use libc directly
• Async I/O: stackless coroutines, io_uring integration
• Testing: built-in test framework, comptime test execution
• SIMD: @Vector(N, T) with comptime-known sizes
• Sentinel-terminated slices: [:0]const u8 for C string interop
• inline for/while: loop unrolling controlled by programmer""",
    "python": """\
== Python Advanced Patterns ==
• asyncio + uvloop: 2-4x event loop speedup (Cython drop-in)
• Protocol (structural subtyping): duck typing with type safety
• TypedDict + dataclasses: typed dicts for external, dataclasses for internal
• __slots__: 40-60% memory reduction for data classes
• functools.cache/lru_cache: memoization with zero effort
• contextlib.asynccontextmanager: async resource management
• PEP 695 type params: def foo[T](x: T) -> T syntax (3.12+)
• match/case (3.10+): structural pattern matching with guards
• TaskGroup (3.11+): structured concurrency, cancel on first error
• ExceptionGroup: multiple concurrent exceptions
• typing.Annotated + Pydantic v2: runtime validation from type hints
• __init_subclass__: metaclass-lite, hook subclass creation
• descriptors: __get__/__set__ for computed attributes
• C extensions: cffi/ctypes for FFI, Cython for hot loops
• GIL-free threading (3.13+): per-interpreter GIL, free-threaded build""",
    "erlang": """\
== Erlang/OTP Advanced Patterns ==
• Supervision trees: one_for_one, one_for_all, rest_for_one strategies
• gen_server: stateful process, handle_call/cast/info callbacks
• gen_statem: modern FSM (replaced gen_fsm), state_functions or handle_event
• Distributed apps: {distributed, [{app, [Node]}]} in kernel config
• Hot code upgrade: code_change/3 callback, release handling
• ETS/DETS: in-memory/disk term storage, concurrent reads
• Process links + monitors: bidirectional crash propagation
• Selective receive: pattern match in receive clause, mailbox scanning
• OTP releases: relx/rebar3, sys.config, vm.args
• Binary pattern matching: <<Size:16, Data:Size/binary>> for protocols
• NIFs: C extensions for CPU-hot paths (careful: blocks scheduler!)
• Port drivers: external process communication via stdin/stdout
• pg (process groups): pub/sub across distributed nodes
• Dialyzer: success typing, -spec annotations, plt analysis""",
    "elixir": """\
== Elixir Advanced Patterns ==
• Phoenix LiveView: real-time UI over WebSocket, no JS needed
• GenServer: stateful process, handle_call/cast/info + init
• Supervisor: child_spec, DynamicSupervisor for runtime children
• PubSub: Phoenix.PubSub for cross-process/cross-node messaging
• Ecto: changesets for validation, multi-tenant with prefix
• Metaprogramming: quote/unquote, __using__ macro, compile-time codegen
• Protocols: polymorphism via defprotocol/defimpl (like Rust traits)
• Streams: lazy enumerables, Stream.resource for external data
• Task.async_stream: bounded concurrency with max_concurrency
• ETS: :ets.new for shared state, concurrent reads
• Oban: persistent job queue, cron scheduling, unique jobs
• Nx + Scholar: numerical computing, ML on BEAM
• LiveView Streams: efficient large list rendering without full re-render
• Telemetry: :telemetry.execute for metrics, LiveDashboard for monitoring
• Releases: mix release, runtime config, Docker-friendly deployment""",
}


# ---------------------------------------------------------------------------
# Toolkit
# ---------------------------------------------------------------------------


class CodingTools(Toolkit):
    """Expertise em programacao avancada — C, Rust, Zig, Python, Erlang, Elixir."""

    def __init__(self, workspace: Path | None = None) -> None:
        super().__init__(name="coding_tools")
        self._workspace = workspace or Path(tempfile.gettempdir())
        self._code_dir = self._workspace / "code"
        self._code_dir.mkdir(parents=True, exist_ok=True)
        self.register(self.code_run)
        self.register(self.code_reference)
        self.register(self.code_languages)
        self.register(self.code_benchmark)

    async def _exec(
        self, cmd: str, timeout: float = 30.0, cwd: str | None = None,
    ) -> tuple[str, str, int]:
        """Run a shell command and return (stdout, stderr, returncode)."""
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout,
        )
        return (
            stdout.decode(errors="replace").strip(),
            stderr.decode(errors="replace").strip(),
            proc.returncode or 0,
        )

    async def code_run(self, language: str, code: str) -> str:
        """Compila e executa um trecho de codigo em qualquer linguagem suportada.

        Suporta: c, rust, zig, python, erlang, elixir.
        O codigo e salvo num arquivo temporario, compilado (se necessario),
        e executado. Retorna stdout + stderr.

        Args:
            language: Linguagem (c, rust, zig, python, erlang, elixir).
            code: Codigo-fonte completo a executar.
        """
        lang = language.lower().strip()
        if lang not in _LANGS:
            return f"Linguagem '{lang}' nao suportada. Use: {', '.join(_LANGS)}"

        spec = _LANGS[lang]
        ext = spec["ext"]

        with tempfile.TemporaryDirectory(prefix="enton_code_", dir=self._code_dir) as tmpdir:
            src = Path(tmpdir) / f"main{ext}"
            out = Path(tmpdir) / "main"
            src.write_text(code, encoding="utf-8")

            # For Erlang, extract module name from code
            module = "main"
            if lang == "erlang":
                for line in code.splitlines():
                    if line.strip().startswith("-module("):
                        module = line.split("(")[1].split(")")[0].strip()
                        src = Path(tmpdir) / f"{module}.erl"
                        src.write_text(code, encoding="utf-8")
                        break

            # Compile step
            compile_cmd = str(spec["compile"])
            if compile_cmd:
                cmd = compile_cmd.format(
                    src=src, out=out, dir=tmpdir, module=module,
                )
                # Check if compiler exists
                compiler = cmd.split()[0]
                if not shutil.which(compiler):
                    return f"Compilador '{compiler}' nao encontrado no sistema."
                cout, cerr, crc = await self._exec(cmd, cwd=tmpdir)
                if crc != 0:
                    return f"ERRO de compilacao ({spec['name']}):\n{cerr or cout}"

            # Run step
            run_cmd = str(spec["run"]).format(
                src=src, out=out, dir=tmpdir, module=module,
            )
            runner = run_cmd.split()[0]
            if not shutil.which(runner) and not Path(runner).exists():
                return f"Runtime '{runner}' nao encontrado."

            rout, rerr, rrc = await self._exec(run_cmd, cwd=tmpdir)
            result = rout
            if rerr:
                result += f"\nSTDERR: {rerr}"
            if rrc != 0:
                result += f"\n(exit code: {rrc})"
            return result or "(sem output)"

    async def code_reference(self, language: str, topic: str = "") -> str:
        """Consulta a base de conhecimento de padroes avancados de uma linguagem.

        Retorna patterns, idioms, e best practices para a linguagem pedida.

        Args:
            language: Linguagem (c, rust, zig, python, erlang, elixir).
            topic: Topico opcional pra filtrar (ex: 'async', 'lock-free', 'supervisor').
        """
        lang = language.lower().strip()
        if lang not in _KNOWLEDGE:
            return f"Sem knowledge pra '{lang}'. Disponiveis: {', '.join(_KNOWLEDGE)}"

        kb = _KNOWLEDGE[lang]
        if topic:
            # Filter lines matching the topic
            topic_lower = topic.lower()
            lines = [
                line for line in kb.splitlines()
                if topic_lower in line.lower()
            ]
            if lines:
                return f"Patterns de {lang} sobre '{topic}':\n" + "\n".join(lines)
            return f"Nenhum pattern de {lang} sobre '{topic}'. Knowledge completo:\n{kb}"
        return kb

    async def code_languages(self) -> str:
        """Lista as linguagens suportadas e seus compiladores/runtimes.

        Args:
            (nenhum)
        """
        lines = []
        for lang_id, spec in _LANGS.items():
            compiler = str(spec.get("compile", "") or spec.get("run", "")).split()[0]
            available = "OK" if shutil.which(compiler) else "NAO INSTALADO"
            lines.append(f"  {lang_id}: {spec['name']} [{available}]")
        return "Linguagens disponiveis:\n" + "\n".join(lines)

    async def code_benchmark(self, language: str, code: str, runs: int = 3) -> str:
        """Compila e executa codigo N vezes, retornando tempo medio de execucao.

        Util pra comparar performance entre linguagens.

        Args:
            language: Linguagem (c, rust, zig, python, erlang, elixir).
            code: Codigo-fonte completo.
            runs: Numero de execucoes (default: 3).
        """
        lang = language.lower().strip()
        if lang not in _LANGS:
            return f"Linguagem '{lang}' nao suportada."

        spec = _LANGS[lang]
        ext = spec["ext"]
        runs = min(runs, 10)  # cap at 10

        with tempfile.TemporaryDirectory(prefix="enton_bench_", dir=self._code_dir) as tmpdir:
            src = Path(tmpdir) / f"main{ext}"
            out = Path(tmpdir) / "main"
            src.write_text(code, encoding="utf-8")

            module = "main"
            if lang == "erlang":
                for line in code.splitlines():
                    if line.strip().startswith("-module("):
                        module = line.split("(")[1].split(")")[0].strip()
                        src = Path(tmpdir) / f"{module}.erl"
                        src.write_text(code, encoding="utf-8")
                        break

            # Compile
            compile_cmd = str(spec["compile"])
            if compile_cmd:
                cmd = compile_cmd.format(
                    src=src, out=out, dir=tmpdir, module=module,
                )
                _, cerr, crc = await self._exec(cmd, cwd=tmpdir)
                if crc != 0:
                    return f"ERRO de compilacao:\n{cerr}"

            # Benchmark
            run_cmd = str(spec["run"]).format(
                src=src, out=out, dir=tmpdir, module=module,
            )
            times: list[float] = []
            last_output = ""
            for _ in range(runs):
                t0 = time.perf_counter()
                rout, _, _ = await self._exec(run_cmd, cwd=tmpdir)
                elapsed = time.perf_counter() - t0
                times.append(elapsed)
                last_output = rout

            avg = sum(times) / len(times)
            best = min(times)
            worst = max(times)
            return (
                f"Benchmark {spec['name']} ({runs} runs):\n"
                f"  media: {avg*1000:.1f}ms\n"
                f"  melhor: {best*1000:.1f}ms\n"
                f"  pior: {worst*1000:.1f}ms\n"
                f"  output: {last_output[:200]}"
            )
