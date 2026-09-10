import time

from utils.timer import Timer, timed


# ============================================================
# 1. Timer.start() / stop()
# ============================================================
timer = Timer(sync_cuda=False)

returned = timer.start()

assert returned is timer
assert timer.start_time is not None

time.sleep(0.05)

elapsed = timer.stop()

assert isinstance(elapsed, float)
assert elapsed >= 0.04
assert timer.start_time is None
assert timer.elapsed == elapsed


# ============================================================
# 2. stop() before start()
# ============================================================
timer = Timer(sync_cuda=False)

elapsed = timer.stop()

assert elapsed == 0.0
assert timer.elapsed == 0.0


# ============================================================
# 3. Timer can be started again
# ============================================================
timer = Timer(sync_cuda=False)

timer.start()
time.sleep(0.02)
first_elapsed = timer.stop()

timer.start()
time.sleep(0.02)
second_elapsed = timer.stop()

assert first_elapsed >= 0.01
assert second_elapsed >= 0.01
assert timer.start_time is None


# ============================================================
# 4. Context manager
# ============================================================
with Timer(sync_cuda=False) as timer:
    time.sleep(0.05)

assert timer.elapsed >= 0.04
assert timer.start_time is None


# ============================================================
# 5. timed() context manager
# ============================================================
with timed(sync_cuda=False) as timer:
    time.sleep(0.05)

assert timer.elapsed >= 0.04
assert timer.start_time is None


# ============================================================
# 6. Exception inside timed() must still stop timer
# ============================================================
timer = None

try:
    with timed(sync_cuda=False) as timer:
        time.sleep(0.02)
        raise RuntimeError("test error")
except RuntimeError:
    pass
else:
    raise AssertionError("Expected RuntimeError was not raised")

assert timer is not None
assert timer.elapsed >= 0.01
assert timer.start_time is None