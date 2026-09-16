from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(
    config_path: Union[str, Path] = "configs/config.yaml",
    experiment_name: Optional[str] = None,
    experiments_path: Union[str, Path] = "configs/experiments.yaml",
) -> Dict[str, Any]:
    cfg = load_yaml(config_path)

    if experiment_name is None:
        return cfg

    experiments = load_yaml(experiments_path)

    # Support both "E2_cnn" and "architecture.E2_cnn"
    exp_cfg = None
    if "." in experiment_name:
        group, name = experiment_name.split(".", 1)
        exp_cfg = experiments.get(group, {}).get(name)
    else:
        # Search all groups
        for group_dict in experiments.values():
            if isinstance(group_dict, dict) and experiment_name in group_dict:
                exp_cfg = group_dict[experiment_name]
                break

    if exp_cfg is None:
        raise KeyError(
            f"Experiment '{experiment_name}' not found in {experiments_path}"
        )

    # Remove description key so it does not pollute the runtime config
    exp_cfg = {k: v for k, v in exp_cfg.items() if k != "description"}

    return deep_merge(cfg, exp_cfg)


def get_device(cfg: Dict[str, Any]):
    import torch

    requested = cfg.get("project", {}).get("device", "cuda").lower()

    if requested == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(requested)


def ensure_dirs(cfg: Dict[str, Any]) -> None:
    logging_cfg = cfg.get("logging", {})
    for key in ("log_dir", "checkpoint_dir", "figure_dir", "result_dir"):
        path = logging_cfg.get(key)
        if path:
            Path(path).mkdir(parents=True, exist_ok=True)