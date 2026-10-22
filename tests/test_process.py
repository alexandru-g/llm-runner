"""Process liveness."""
import subprocess
import time
from pathlib import Path

from llm_runner import process as proc


def test_a_crashed_child_is_not_alive():
    """`run --wait` is the server's parent, so a crashed server is a zombie
    until llmctl exits; it must still count as dead."""
    child = subprocess.Popen(["sh", "-c", "exit 3"])
    deadline = time.time() + 5
    while proc._alive(child.pid) and time.time() < deadline:
        time.sleep(0.05)
    assert not proc._alive(child.pid)          # dead, though not reaped yet
    assert child.wait() == 3


def test_a_running_process_is_alive():
    child = subprocess.Popen(["sleep", "5"])
    try:
        assert proc._alive(child.pid)
    finally:
        child.kill()
        child.wait()


def test_wait_fails_fast_with_the_log_tail(tmp_path, capsys):
    """A server that dies while loading: `run --wait` reaps it and prints why."""
    from llm_runner import cli
    log = tmp_path / "m.log"
    log.write_text("".join(f"line {i}\n" for i in range(100)) + "ImportError: boom\n")
    child = subprocess.Popen(["sh", "-c", "exit 1"])
    rec = proc.RunningProcess(name="m", pid=child.pid, port=1, started_at=0, model_path="m",
                              log_file=str(log), cmd=[], backend="vllm")
    start = time.time()
    assert cli._wait_healthy(rec, timeout=30) is False
    assert time.time() - start < 10
    err = capsys.readouterr().err
    assert "ImportError: boom" in err and "line 99" in err and "line 50" not in err
    assert not Path(f"/proc/{child.pid}").exists()      # reaped, not a zombie
