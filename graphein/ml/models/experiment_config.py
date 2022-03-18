import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

import yaml
from pydantic import BaseModel

from .model_config import ModelConfig, parse_model_config

log = logging.getLogger(__name__)

DEFAULT_CONFIG: Dict[str, Any] = {
    "task": ["train"],  # ["train", "val", "test", "inference"]
    "device": "cpu",
    "precision": "mixed",
    "seed": 42,
    "dataloader": {
        "cached": False,
        "batch_size": 16,
        "batch_type": "item",
        "shuffle": True,
        "data_usage_ratio": 1.0,
        "multi_process_count": 0,
        "multi_process_max_pre_fetch": 2,
    },
    "model": {
        "type": "graph_classifier",
        "graph_encoder": {},
        "classifier": {
            "layer_dims": [128, 2],
            "activations": ["relu", "sigmoid"],
            "layer_norm": [
                True,
                True,
            ],
            "batch_norm": [False, False],
            "dropout": 0.3,
        },
        "loss": {"type": "cross_entropy", "label_smoothing": 0.0},
    },
    "optimiser": {
        "type": "adam",  # Todo setup ADAM config with betas
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 0.000000009,
        "learning_rate": 0.0001,
        "weight_decay": 0.0,
        "amsgrad": False,
    },
}


class Device(Enum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


class Precision(Enum):
    FULL = "full"
    MIXED = "mixed"


class Task(Enum):
    TRAIN = "train"
    EVALUATE = "val"
    TEST = "test"


class TrainerConfig(BaseModel):
    """
    Configuration for a trainer.
    """

    num_epochs: int
    seed: int
    device: Device
    precision: Precision


class ExperimentConfig(BaseModel):
    tasks: List[Task]
    # dataloader: DataLoaderConfig
    model: ModelConfig
    trainer: TrainerConfig

    class Config:
        arbitrary_types_allowed: bool = True


def parse_trainer_config(config: Dict[str, Any]) -> TrainerConfig:
    config = config["trainer"]
    return TrainerConfig(
        num_epochs=config["num_epochs"],
        device=Device(config["device"]),
        seed=config["seed"],
        precision=Precision(config["precision"]),
    )


def load_config(file_path: str) -> ExperimentConfig:
    """
    Load and parse the config file from a filepath.

    :param file_path: Path to the YAML/JSON config file.
    :type file_path: str
    :return: The parsed config.
    :rtype: ExperimentConfig
    """
    log.info(f"Loading config '{file_path}'")
    path = Path(file_path)
    with path.open() as file:
        config = yaml.safe_load(file)
        return parse_config(config)


def parse_config(config: Dict[str, Any]) -> ExperimentConfig:
    """
    Parse the root from an untyped dictionary containing configuration options.

    :param config: The config dictionary to parse.
    :type config: Dict[str, Any]
    :return: The parsed experiment config object.
    :rtpe: ExperimentConfig
    """

    return ExperimentConfig(
        tasks=[Task(t) for t in config["task"]],
        # dataloader=parse_dataloader_config(config),
        model=parse_model_config(config),
        trainer=parse_trainer_config(config),
        # aux_tasks=[AuxTask(task) for task in config["aux_tasks"]],
        # evaluate=parse_evaluation_config(config),
    )
