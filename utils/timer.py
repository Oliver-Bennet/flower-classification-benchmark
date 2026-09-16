from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator, Optional

import torch


class Timer:

    def __init__(self, sync_cuda: bool = True):
        self.sync_cuda = sync_cuda and torch.cuda.is_available()
        self.start_time: Optional[float] = None
        self.elapsed: float = 0.0

    def _sync(self):
        if self.sync_cuda:
            torch.cuda.synchronize()

    def start(self) -> "Timer":
        self._sync()
        self.start_time = time.perf_counter()
        return self

    def stop(self) -> float:
        self._sync()
        if self.start_time is None:
            return 0.0
        self.elapsed = time.perf_counter() - self.start_time
        self.start_time = None
        return self.elapsed

    def __enter__(self) -> "Timer":
        return self.start()

    def __exit__(self, *args) -> None:
        self.stop()


@contextmanager
def timed(sync_cuda: bool = True) -> Generator[Timer, None, None]:
    t = Timer(sync_cuda=sync_cuda)
    t.start()
    try:
        yield t
    finally:
        t.stop()