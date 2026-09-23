"""Run a child process with a timeout that actually stops it.

Both run_bash and run_python hand the model's code to a subprocess with a timeout, so
a hung command can't freeze the agent. The obvious way to do that —
`subprocess.run(..., timeout=N)` — has a hole: on timeout it kills only the direct
child. For run_bash the direct child is /bin/sh; whatever sh started (a `sleep`, a
server, a stuck build) is a *grandchild*, and it keeps running. The tool then reports
"command timed out" while the command is, in fact, still going. A timeout that
unfreezes the agent but not the machine is a false promise.

The fix is to run the child in its own process group (`start_new_session=True`, which
calls setsid() in the child) and, on timeout, signal the whole group with
os.killpg(). Every descendant that hasn't deliberately left the group dies with it —
which is exactly what "stop this command" should mean.

One thing killpg cannot reach: a descendant that *did* leave the group (setsid(), the
double-fork a daemon does, a server started with nohup). That process is the model's
doing and outlives the kill — but it also inherited the tool's stdout/stderr pipe, and
as long as it lives the pipe never reaches EOF. So after the kill we drain the pipes
for a bounded moment rather than "until EOF"; otherwise a timed-out command that
started a daemon would freeze the agent for the daemon's whole lifetime, which is the
very thing the timeout exists to prevent.

The helper returns the decoded output either way; a timeout comes back as
subprocess.TimeoutExpired after the group has been killed, so callers can format the
error however they like. That exception carries whatever the command printed before
it was stopped (see partial_output): a test that hung prints its name first, a script
that stalled usually printed why, and throwing that away would leave the model with
nothing but "timed out" to reason from.
"""

from __future__ import annotations

import os
import signal
import subprocess

# How long to keep reading the pipes after the process group has been killed. In the
# normal case the read returns at once: every writer is dead, so the pipes hit EOF.
# The exception is a descendant that left the group (see the module docstring) — it
# still holds the pipes open, so an unbounded read would block for its whole
# lifetime. One second is ample for the kernel to hand over what the dead writers
# had already written.
PIPE_DRAIN_TIMEOUT = 1.0  # seconds


def run_with_timeout(
    args,
    *,
    cwd,
    timeout: float,
    shell: bool = False,
) -> subprocess.CompletedProcess:
    """Run `args` (a list, or a command string when shell=True) and capture its output.

    Raises subprocess.TimeoutExpired if it exceeds `timeout` seconds — but only after
    the child *and everything it spawned* have been killed. The exception's .stdout
    and .stderr hold what the command printed before the kill — as text, or as raw
    bytes / None if a descendant escaped the kill and the pipes had to be abandoned
    (partial_output() normalizes both).
    """
    proc = subprocess.Popen(
        args,
        shell=shell,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,   # the child leads a fresh process group (POSIX)
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _kill_process_group(proc)
        # Reap the child and release the pipes. This second communicate() also hands
        # back everything the command printed before it was killed — Python promises
        # that retrying communicate() after a timeout loses no output — decoded as
        # text, where the exception itself only carried raw bytes. Attach it so the
        # tool can show the model what happened up to the hang.
        #
        # The read is bounded, not "until EOF": a descendant that left the process
        # group survived the kill and still holds the pipes open (module docstring),
        # and waiting for it would freeze the agent exactly as the original hang did.
        try:
            exc.stdout, exc.stderr = proc.communicate(timeout=PIPE_DRAIN_TIMEOUT)
        except subprocess.TimeoutExpired as late:
            # EOF is not coming. Settle for what was captured so far — raw bytes on
            # this path, which partial_output() decodes — reap the killed child (it
            # is already dead, so this returns at once) and close our ends of the
            # pipes. The escaped process itself is left alone: we cannot find it
            # portably, and it is the command's doing, not the tool's.
            exc.stdout, exc.stderr = late.stdout, late.stderr
            proc.wait()
            for pipe in (proc.stdout, proc.stderr):
                if pipe is not None:
                    pipe.close()
        raise
    return subprocess.CompletedProcess(args, proc.returncode, stdout, stderr)


def partial_output(exc: subprocess.TimeoutExpired) -> tuple[str, str]:
    """The (stdout, stderr) a timed-out command printed before it was killed, as text.

    run_with_timeout normally attaches decoded text, but when it had to abandon the
    pipes (a descendant escaped the kill) it attaches the raw bytes — or None if
    nothing was printed — and a TimeoutExpired raised elsewhere may carry either. All
    three shapes are normalized here rather than in every tool.
    """
    def as_text(value) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value

    return as_text(exc.stdout), as_text(exc.stderr)


def _kill_process_group(proc: subprocess.Popen) -> None:
    """SIGKILL the child's whole process group, not just the child."""
    if not hasattr(os, "killpg"):
        proc.kill()               # Windows has no process groups in this sense
        return
    try:
        # With start_new_session=True the group id equals the child's pid.
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass                      # every member already exited on its own
