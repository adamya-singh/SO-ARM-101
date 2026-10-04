"""``ChunkEnv`` workers in spawned processes, stepped asynchronously (retry tranche, 2026-10-04).

The legacy ``ACTSubprocVecEnv`` pattern (spawn, pipes, one persistent simulator per worker) without its
gym layer. Workers import only the simulator (no torch: about 400 MB less per worker on a 15 GB host).
``send``/``recv`` let the collector keep the GPU busy on one group of environments while the others
simulate their chunk.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import traceback
from multiprocessing.connection import Connection, wait
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _worker(remote: Connection, model_path: str, reward_kwargs: dict, closure: bool) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    env = None
    try:
        from so_arm101_v2.rl.env import ChunkEnv
        from so_arm101_v2.rl.reward import RewardConfig
        env = ChunkEnv(model_path, RewardConfig(**reward_kwargs), closure=closure)
        remote.send(("ready", dict(torch_imported="torch" in sys.modules, pid=os.getpid())))
        while True:
            command, payload = remote.recv()
            if command == "reset":
                scenario, frame = payload
                remote.send(("ok", env.reset(scenario, frame=frame)))
            elif command == "step":
                commands, frame = payload
                remote.send(("ok", env.step_chunk(commands, frame=frame)))
            elif command == "_stall":         # tests only: a worker that never answers (the EGL hang of 2026-10-04)
                import time
                time.sleep(10 ** 6)
            elif command == "close":
                remote.send(("ok", None))
                break
            else:
                raise ValueError(f"unknown command {command!r}")
    except BaseException:
        try:
            remote.send(("error", traceback.format_exc()))
        except Exception:
            pass
    finally:
        if env is not None:
            env.close()
        remote.close()


class WorkerError(RuntimeError):
    pass


class SubprocChunkVecEnv:
    def __init__(self, model_path: str | Path, num_envs: int, *, reward_kwargs: dict | None = None, closure: bool = True) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self._ctx = mp.get_context("spawn")
        self._args = (str(Path(model_path).resolve()), dict(reward_kwargs or {}), closure)
        self.remotes: list[Connection] = []
        self.processes = []
        for _ in range(num_envs):
            parent, process = self._spawn()
            self.remotes.append(parent)
            self.processes.append(process)
        self.info = [self._receive(i) for i in range(num_envs)]
        self.pending: set[int] = set()
        self.sent_at: dict[int, float] = {}
        self.restarts = 0

    def _spawn(self):
        parent, child = self._ctx.Pipe()
        process = self._ctx.Process(target=_worker, args=(child, *self._args), daemon=True)
        process.start()
        child.close()
        return parent, process

    def restart(self, index: int) -> None:
        """Kill a worker that stopped answering and start a fresh one in its place (its episode is lost)."""
        try:
            self.processes[index].kill()
            self.processes[index].join(timeout=10)
        except Exception:
            pass
        try:
            self.remotes[index].close()
        except Exception:
            pass
        self.remotes[index], self.processes[index] = self._spawn()
        self.info[index] = self._receive(index)
        self.pending.discard(index)
        self.sent_at.pop(index, None)
        self.restarts += 1

    def stalled(self, timeout_s: float) -> list[int]:
        import time
        now = time.time()
        return sorted(i for i in self.pending if now - self.sent_at.get(i, now) > timeout_s)

    @property
    def num_envs(self) -> int:
        return len(self.remotes)

    def _receive(self, index: int) -> Any:
        try:
            status, payload = self.remotes[index].recv()
        except EOFError as exc:
            raise WorkerError(f"env worker {index} died (exit code {self.processes[index].exitcode})") from exc
        if status == "error":
            raise WorkerError(f"env worker {index} failed:\n{payload}")
        return payload

    def send_reset(self, index: int, scenario: Any, *, frame: bool = True) -> None:
        self.remotes[index].send(("reset", (scenario, frame)))
        self.pending.add(index)
        self.sent_at[index] = __import__("time").time()

    def send_step(self, index: int, commands: np.ndarray, *, frame: bool = True) -> None:
        self.remotes[index].send(("step", (np.asarray(commands, np.float32), frame)))
        self.pending.add(index)
        self.sent_at[index] = __import__("time").time()

    def recv(self, index: int) -> Any:
        payload = self._receive(index)
        self.pending.discard(index)
        return payload

    def ready(self, timeout: float | None = None) -> list[int]:
        """Indices of pending environments whose answer has arrived (blocks until at least one, or ``timeout``)."""
        if not self.pending:
            return []
        by_remote = {self.remotes[i]: i for i in self.pending}
        return sorted(by_remote[c] for c in wait(list(by_remote), timeout=timeout))

    def close(self) -> None:
        for index, remote in enumerate(self.remotes):
            try:
                if index in self.pending:
                    remote.recv()
                remote.send(("close", None))
                remote.recv()
            except Exception:
                pass
        for process in self.processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()

    def __enter__(self) -> "SubprocChunkVecEnv":
        return self

    def __exit__(self, *exc: Iterable[Any]) -> None:
        self.close()


__all__ = ["SubprocChunkVecEnv", "WorkerError"]
