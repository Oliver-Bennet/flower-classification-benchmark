from engine.early_stopping import EarlyStopping


# ============================================================
# 1. First score -> establishes best, does not stop
# ============================================================
es = EarlyStopping(patience=3, min_delta=0.0, mode="max")

assert es(0.50, epoch=1) is False
assert es.best_score == 0.50
assert es.best_epoch == 1
assert es.counter == 0
assert es.should_stop is False


# ============================================================
# 2. Improvement -> best updates and counter resets
# ============================================================
assert es(0.60, epoch=2) is False

assert es.best_score == 0.60
assert es.best_epoch == 2
assert es.counter == 0
assert es.should_stop is False


# ============================================================
# 3. No improvement -> counter increases
# ============================================================
assert es(0.59, epoch=3) is False
assert es.counter == 1

assert es(0.58, epoch=4) is False
assert es.counter == 2


# ============================================================
# 4. patience reached -> should_stop=True
# ============================================================
assert es(0.57, epoch=5) is True

assert es.counter == 3
assert es.should_stop is True
assert es.best_score == 0.60
assert es.best_epoch == 2


# ============================================================
# 5. min mode
# ============================================================
es = EarlyStopping(patience=2, min_delta=0.0, mode="min")

assert es(1.00, epoch=1) is False
assert es(0.80, epoch=2) is False

assert es.best_score == 0.80
assert es.best_epoch == 2
assert es.counter == 0

assert es(0.90, epoch=3) is False
assert es.counter == 1

assert es(1.00, epoch=4) is True
assert es.counter == 2
assert es.should_stop is True


# ============================================================
# 6. min_delta must be respected
# ============================================================
es = EarlyStopping(
    patience=2,
    min_delta=0.01,
    mode="max",
)

assert es(0.50, epoch=1) is False

# +0.005 < min_delta -> NOT improvement
assert es(0.505, epoch=2) is False
assert es.best_score == 0.50
assert es.counter == 1

# +0.02 > min_delta -> improvement
assert es(0.52, epoch=3) is False
assert es.best_score == 0.52
assert es.best_epoch == 3
assert es.counter == 0


# ============================================================
# 7. reset()
# ============================================================
es = EarlyStopping(patience=2)

es(0.50, epoch=1)
es(0.40, epoch=2)

assert es.best_score == 0.50
assert es.counter == 1

es.reset()

assert es.counter == 0
assert es.best_score is None
assert es.should_stop is False
assert es.best_epoch is None


# ============================================================
# 8. Invalid mode
# ============================================================
try:
    EarlyStopping(mode="invalid")
except ValueError:
    pass
else:
    raise AssertionError("Invalid mode should raise ValueError")


# ============================================================
# 9. state_dict / load_state_dict
# ============================================================
es = EarlyStopping(
    patience=3,
    min_delta=0.01,
    mode="max",
)

es(0.50, epoch=1)
es(0.60, epoch=2)
es(0.59, epoch=3)

state = es.state_dict()

new_es = EarlyStopping(
    patience=99,
    min_delta=99.0,
    mode="min",
)

new_es.load_state_dict(state)

assert new_es.counter == es.counter
assert new_es.best_score == es.best_score
assert new_es.should_stop == es.should_stop
assert new_es.best_epoch == es.best_epoch

# Nếu đã sửa load_state_dict() như đề xuất:
assert new_es.patience == es.patience
assert new_es.min_delta == es.min_delta
assert new_es.mode == es.mode