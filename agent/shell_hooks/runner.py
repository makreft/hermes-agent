"""One hook run: the single subprocess site (``_spawn``), the callback ``invoke_hook()`` calls per
firing, and the failure semantics that turn a run into the hook's contribution."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import replace
from typing import Any, Callable, Optional

# split_command_line, not shlex: shlex eats Windows path backslashes.
from hermes_cli._subprocess_compat import IS_WINDOWS, kill_process_tree, split_command_line, windows_hide_flags

from agent.shell_hooks.spec import _BLOCKING_EVENTS, _TOOL_EVENTS, MAX_TIMEOUT_SECONDS, ShellHookSpec
from agent.shell_hooks.wire import _DEFAULT_BLOCK_MESSAGE, _is_json_object, _parse_response, _serialize_payload

logger = logging.getLogger(__name__)

# Exit code that signals "block this action" independent of stdout (Claude Code / Cursor).
BLOCK_EXIT_CODE = 2

_STDERR_MESSAGE_LIMIT = 400

# Popen failure -> diagnostic; WinError 193 gets its own below, everything else is str(exc).
_POPEN_ERRORS = ((FileNotFoundError, "command not found"), (PermissionError, "command not executable"))

# A hook configured as a bare script path runs on POSIX because the kernel reads its shebang.
# CreateProcess has no such mechanism and answers WinError 193 ("%1 is not a valid Win32
# application") for a text file, so every ``command: "~/.hermes/agent-hooks/x.sh"`` example in
# the hooks docs — the canonical shape — fails on Windows while the same config works everywhere
# else. Map the suffixes that shape uses to their interpreter; unmapped suffixes keep the OS
# failure so a typo still reads as "command not found" rather than a mystery interpreter error.
_WINDOWS_SCRIPT_INTERPRETERS = {".sh": "bash", ".bash": "bash", ".py": "python"}

# WinError 193 raised for a suffix we deliberately do not map.
_NOT_DIRECTLY_EXECUTABLE = "cannot be run directly on Windows (there is no shebang support): start it with its interpreter, e.g. 'bash <path>'"


def _windows_script_argv(argv: list[str]) -> list[str]:
    """``argv`` with the interpreter prepended when element 0 is an existing script we can name an
    interpreter for; unchanged otherwise, including on POSIX, where the shebang already works."""
    suffix = os.path.splitext(argv[0])[1].lower()
    kind = _WINDOWS_SCRIPT_INTERPRETERS.get(suffix)
    if kind is None or not os.path.isfile(argv[0]):
        return argv
    if kind == "python":
        return [sys.executable, *argv]
    # Resolved inside the caller's try: no Git for Windows raises RuntimeError carrying the
    # installer's own actionable guidance, which is a better diagnostic than any we could add.
    from tools.environments.local import _find_bash
    return [_find_bash(), *argv]


def _spawn(spec: ShellHookSpec, stdin_json: str, on_spawn: Optional[Callable[[subprocess.Popen], None]] = None,
           output_limit: Optional[int] = None) -> dict[str, Any]:
    """The single subprocess site: run ``spec.command`` with ``stdin_json`` on stdin. Same result keys for every outcome.
    ``on_spawn`` receives the live process (an ``async_rewake`` run keeps it to stop the hook early);
    ``output_limit`` keeps only the newest characters of each stream (a background hook may print for hours)."""
    result: dict[str, Any] = {"returncode": None, "stdout": "", "stderr": "", "timed_out": False, "elapsed_seconds": 0.0, "error": None}

    def failed(error: str) -> dict[str, Any]:
        result["error"] = error
        return result

    try:
        argv = split_command_line(os.path.expanduser(spec.command))
    except ValueError as exc:
        return failed(f"command {spec.command!r} cannot be parsed: {exc}")
    if not argv:
        return failed("empty command")
    from tools.hosted_tool_policy import host_feature_refusal
    if refusal := host_feature_refusal("shell_hooks", "A shell hook"):  # the one subprocess site
        return failed(refusal)
    t0 = time.monotonic()
    # Own process group on POSIX so a timed-out hook's descendants are reaped with it (Windows: kill_process_tree
    # / taskkill /T). Hooks that finish in time keep detached helpers alive.
    # detached-spawn: ok — awaited hook; its timeout kills the whole group (kill_process_tree)
    popen_kwargs: dict[str, Any] = {"creationflags": windows_hide_flags()} if IS_WINDOWS else {"process_group": 0}
    # HERMES_HOME follows the routed profile (the import-time environ holds the launch profile's), and
    # under multiplexing os.environ carries the DEFAULT profile's secrets, which a secondary's hook
    # script must not inherit; single-profile runs keep the process env byte-for-byte as before.
    from agent.secret_scope import is_multiplex_active
    from tools.environments.local import build_subprocess_env
    try:
        if IS_WINDOWS:
            argv = _windows_script_argv(argv)
        proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, encoding='utf-8', errors='replace', shell=False,
                                env=build_subprocess_env(scrub_secrets=is_multiplex_active()), **popen_kwargs)
    except Exception as exc:
        for cls, msg in _POPEN_ERRORS:
            if isinstance(exc, cls):
                return failed(msg)
        if getattr(exc, "winerror", None) == 193:
            # Unmapped suffix (.zsh, .fish, .rb, …) — the raw WinError text is localized, so an
            # operator on a non-English Windows could not act on it at all.
            return failed(f"{argv[0]!r} {_NOT_DIRECTLY_EXECUTABLE}")
        return failed(str(exc))
    if on_spawn is not None:
        on_spawn(proc)
    try:
        if output_limit is None:
            stdout, stderr = proc.communicate(input=stdin_json, timeout=spec.timeout)
        else:
            stdout, stderr = _communicate_tail(proc, stdin_json, spec.timeout, output_limit)
    except BaseException as exc:
        # BaseException: the hook leads its own process group, so Ctrl+C's SIGINT never reaches it — only we can.
        kill_process_tree(proc)  # the whole tree — forked helpers holding the pipes would stall the drain
        with suppress(Exception):  # the tail readers drain on their own: only wait for the exit
            if output_limit is None:
                proc.communicate(timeout=1)
            else:
                proc.wait(timeout=1)
        if not isinstance(exc, Exception):
            raise
        if not isinstance(exc, subprocess.TimeoutExpired):  # pragma: no cover — defensive
            return failed(str(exc))
        result.update(timed_out=True, elapsed_seconds=round(time.monotonic() - t0, 3))
        return result
    result.update(returncode=proc.returncode, stdout=stdout or "", stderr=stderr or "", elapsed_seconds=round(time.monotonic() - t0, 3))
    return result


def _communicate_tail(proc: subprocess.Popen, stdin_json: str, timeout: float, limit: int) -> tuple[str, str]:
    """``communicate()`` that keeps only the newest ``limit`` characters of stdout and stderr. Raises
    ``TimeoutExpired`` like ``communicate``; the caller kills the tree, which ends the readers. Reads
    raw bytes as they arrive (a text ``read(n)`` waits for n characters, so a short last message
    would still sit in it when a helper the hook started keeps the pipe open)."""
    from agent.memory_provider import spawn_context_thread
    tails = {"stdout": b"", "stderr": b""}
    keep = limit * 4  # bytes: enough for ``limit`` characters of UTF-8

    def feed() -> None:
        with suppress(OSError):  # a hook that never reads its payload closes the pipe on exit
            proc.stdin.write(stdin_json)
            proc.stdin.close()

    def drain(name: str) -> None:
        fd = getattr(proc, name).fileno()
        for chunk in iter(lambda: os.read(fd, 65536), b""):
            tails[name] = (tails[name] + chunk)[-keep:]

    threads = [spawn_context_thread(feed, name="shell-hook-stdin"),
               *(spawn_context_thread(drain, name=f"shell-hook-{n}", args=(n,)) for n in ("stdout", "stderr"))]
    for thread in threads:
        thread.start()
    proc.wait(timeout=timeout)
    for thread in threads:
        thread.join(timeout=1)
    out, err = (tails[n].decode("utf-8", errors="replace")[-limit:] for n in ("stdout", "stderr"))
    return out, err


def _make_callback(spec: ShellHookSpec) -> Callable[..., Optional[dict[str, Any]]]:
    """Build the closure that ``invoke_hook()`` will call per firing."""

    def _callback(**kwargs: Any) -> Optional[dict[str, Any]]:
        from agent.safe_worker_policy import safe_worker_enabled
        from tools.hosted_tool_policy import host_feature_enabled

        # A hosted scope's events carry the tenant's payload; a shell hook is a process on this host.
        if safe_worker_enabled() or not host_feature_enabled("shell_hooks"):
            return None
        if spec.event in _TOOL_EVENTS and not spec.matches_tool(kwargs.get("tool_name")):
            return None
        if spec.async_rewake:  # never waits, never directs: the event goes on at once
            from agent.shell_hooks.rewake import start
            start(spec, getattr(_callback, "shell_hook_key", None), kwargs, _serialize_payload(spec.event, kwargs))
            return None
        return _evaluate_result(spec, _spawn(spec, _serialize_payload(spec.event, kwargs)))

    _callback.__name__ = _callback.__qualname__ = f"shell_hook[{spec.event}:{spec.command}]"
    return _callback


def _fail_closed_block(spec: ShellHookSpec, reason: str) -> dict[str, Any]:
    return {"action": "block", "message": f"hook {spec.command} failed closed: {reason}"}


def _evaluate_result(
    spec: ShellHookSpec, r: dict[str, Any],
) -> Optional[dict[str, Any]]:
    """Turn a :func:`_spawn` diagnostic dict into the hook's contribution.

    Single place that encodes the failure semantics:

    * spawn error / timeout — fail open (log + ``None``) unless the spec
      is ``fail_closed`` on a blocking-capable event, in which case a
      canonical block shape is returned;
    * exit code 2 on a blocking-capable event — block, with the message
      taken from stdout block JSON, then stderr, then a default
      (Claude-Code / Cursor compatible);
    * other non-zero exits — warn, then parse stdout normally; a
      ``fail_closed`` hook blocks if no directive was produced;
    * non-JSON / unparseable stdout on a ``fail_closed`` blocking hook —
      block instead of silently contributing nothing.

    Shared by the live callback path (:func:`_make_callback`) and the CLI
    test helper (:func:`run_once`) so ``hermes hooks test`` reflects
    production behaviour exactly.
    """
    blocking_event = spec.event in _BLOCKING_EVENTS
    fail_closed = spec.fail_closed and blocking_event
    if r["error"]:
        logger.warning("shell hook failed (event=%s command=%s): %s", spec.event, spec.command, r["error"])
    elif r["timed_out"]:
        logger.warning("shell hook timed out after %.2fs (event=%s command=%s)", r["elapsed_seconds"], spec.event, spec.command)
    if r["error"] or r["timed_out"]:
        return _fail_closed_block(spec, r["error"] or f"timed out after {spec.timeout}s") if fail_closed else None
    stderr = r["stderr"].strip()
    if stderr:
        logger.debug("shell hook stderr (event=%s command=%s): %s", spec.event, spec.command, stderr[:_STDERR_MESSAGE_LIMIT])
    if r["returncode"] == BLOCK_EXIT_CODE and blocking_event:
        parsed = _parse_response(spec.event, r["stdout"])
        if isinstance(parsed, dict) and parsed.get("action") == "block":
            return parsed
        message = stderr[:_STDERR_MESSAGE_LIMIT] or _DEFAULT_BLOCK_MESSAGE
        logger.info("shell hook exited %d — blocking (event=%s command=%s): %s", BLOCK_EXIT_CODE, spec.event, spec.command, message)
        return {"action": "block", "message": message}
    # Other non-zero exits: still parse stdout so exit-code failures can carry a block directive.
    if r["returncode"] != 0:
        logger.warning("shell hook exited %d (event=%s command=%s); stderr=%s",
                       r["returncode"], spec.event, spec.command, stderr[:_STDERR_MESSAGE_LIMIT])
    stdout = (r["stdout"] or "").strip()
    parsed = _parse_response(spec.event, stdout)
    if parsed is None and fail_closed and r["returncode"] != 0:
        return _fail_closed_block(
            spec, f"hook exited {r['returncode']} with no directive",
        )

    if parsed is None and fail_closed and stdout and not _is_json_object(stdout):
        # The hook produced output we could not turn into a directive.
        # A fail-closed gate must not silently allow the action on
        # garbage output (e.g. a stack trace on stdout).
        return _fail_closed_block(spec, "unparseable stdout (expected a JSON object)")
    return parsed


def run_once(spec: ShellHookSpec, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Fire one hook with a synthetic payload (``hermes hooks test`` / doctor) through the production path.
    An ``async_rewake`` hook runs in the foreground here, capped at the synchronous maximum, and reports
    the wake it would deliver."""
    if spec.async_rewake:
        from agent.shell_hooks.rewake import WAKE_EXIT_CODE, wake_output
        result = _spawn(replace(spec, timeout=min(spec.timeout, MAX_TIMEOUT_SECONDS)), _serialize_payload(spec.event, kwargs))
        result["parsed"] = {"wake": wake_output(result) if result["returncode"] == WAKE_EXIT_CODE else None}
        return result
    result = _spawn(spec, _serialize_payload(spec.event, kwargs))
    result["parsed"] = _evaluate_result(spec, result)
    return result
