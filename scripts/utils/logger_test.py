import csv
import json
import tempfile
from pathlib import Path

from utils.logger import TrainingLogger


# ============================================================
# 1. Create logger
# ============================================================
with tempfile.TemporaryDirectory() as tmp:
    log_dir = Path(tmp)

    logger = TrainingLogger(
        log_dir=log_dir,
        experiment_name="test_run",
        console=False,
    )

    assert logger.csv_path == log_dir / "test_run_history.csv"
    assert logger.json_path == log_dir / "test_run_history.json"
    assert logger.history == []


    # ========================================================
    # 2. Log multiple epochs
    # ========================================================
    logger.log(
        epoch=1,
        metrics={
            "train_loss": 1.2,
            "val_accuracy": 0.70,
            "val_macro_f1": 0.65,
        },
    )

    logger.log(
        epoch=2,
        metrics={
            "train_loss": 0.9,
            "val_accuracy": 0.80,
            "val_macro_f1": 0.75,
        },
    )

    logger.log(
        epoch=3,
        metrics={
            "train_loss": 0.7,
            "val_accuracy": 0.78,
            "val_macro_f1": 0.82,
        },
    )

    assert len(logger.history) == 3
    assert logger.history[0]["epoch"] == 1
    assert logger.history[1]["val_accuracy"] == 0.80
    assert logger.history[2]["val_macro_f1"] == 0.82


    # ========================================================
    # 3. CSV file must exist and contain all rows
    # ========================================================
    assert logger.csv_path.exists()

    with open(
        logger.csv_path,
        "r",
        newline="",
        encoding="utf-8",
    ) as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 3

    assert rows[0]["epoch"] == "1"
    assert rows[1]["epoch"] == "2"
    assert rows[2]["epoch"] == "3"

    assert float(rows[1]["val_accuracy"]) == 0.80
    assert float(rows[2]["val_macro_f1"]) == 0.82


    # ========================================================
    # 4. get_best() - max
    # ========================================================
    best_accuracy = logger.get_best(
        key="val_accuracy",
        mode="max",
    )

    assert best_accuracy["epoch"] == 2
    assert best_accuracy["val_accuracy"] == 0.80


    best_f1 = logger.get_best(
        key="val_macro_f1",
        mode="max",
    )

    assert best_f1["epoch"] == 3
    assert best_f1["val_macro_f1"] == 0.82


    # ========================================================
    # 5. get_best() - min
    # ========================================================
    best_loss = logger.get_best(
        key="train_loss",
        mode="min",
    )

    assert best_loss["epoch"] == 3
    assert best_loss["train_loss"] == 0.7


    # ========================================================
    # 6. Invalid mode
    # ========================================================
    try:
        logger.get_best(
            key="val_accuracy",
            mode="invalid",
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Invalid mode should raise ValueError"
        )


    # ========================================================
    # 7. Empty logger
    # ========================================================
    empty_logger = TrainingLogger(
        log_dir=log_dir,
        experiment_name="empty",
        console=False,
    )

    assert empty_logger.get_best() == {}


    # ========================================================
    # 8. save_json()
    # ========================================================
    logger.save_json()

    assert logger.json_path.exists()

    with open(
        logger.json_path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    assert data == logger.history
    assert len(data) == 3
    assert data[0]["epoch"] == 1
    assert data[2]["val_macro_f1"] == 0.82


    # ========================================================
    # 9. close() must close CSV file
    # ========================================================
    logger.close()

    assert logger._csv_file is None
    assert logger._csv_writer is None

    # JSON must still exist after close
    assert logger.json_path.exists()