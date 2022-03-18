from enum import Enum
from typing import Any, Dict, List, NamedTuple, Optional

import pydantic
from pydantic import BaseModel


class ModuleConfigurationError(Exception):
    """Error that is raised when a model submodule is incorrectly configured in the config"""

    def __init__(self, message):
        self.message = message
        super().__init__(message)


class ModelConfigurationError(Exception):
    """Error that is raised the model is incorrectly configured in the config"""

    def __init__(self, message):
        self.message = message
        super().__init__(message)


class ModelType(Enum):
    NODE_CLASSIFIER = "node_classifier"
    NODE_REGRESSION = "node_regression"
    GRAPH_CLASSIFIER = "graph_classifier"
    GRAPH_REGRESSION = "graph_regression"


class ActivationType(Enum):
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"
    TANH = "tanh"
    RELU = "relu"
    ELU = "elu"
    MISH = "mish"
    SWISH = "swish"
    LEAKY_RELU = "leaky_relu"
    NONE = "none"


class LayerType(Enum):
    GAT = "gat"
    GCN = "gcn"
    LINEAR = "linear"
    MPNN = "mpnn"
    GNN_FILM = "gnn_film"


class AggregationType(Enum):
    SUM = "sum"
    MEAN = "mean"
    WEIGHTED_SUM = "weighted_sum"
    SOFT_ATTENTION = "soft_attention"


class LossType(Enum):
    CROSS_ENTROPY = "cross_entropy"
    LABEL_SMOOTHING = "label_smoothing"
    MSE = "mse"
    BCE = "bce"
    WEIGHTED_BCE = "weighted_bce"
    BCE_LOGITS = "bce_logits"


class OptimiserType(Enum):
    SGD = "sgd"
    ADAM = "adam"


class ActivationConfig(NamedTuple):
    type: ActivationType
    kwargs: Any


class LossConfig(BaseModel):
    type: LossType
    label_smoothing: float


class ClassifierConfig(BaseModel):
    # pretrained: Optional[str]
    layer_dims: List[int]
    activations: List[ActivationType]
    layer_norm: List[bool]
    batch_norm: List[bool]
    dropout: float

    class Config:
        arbitrary_types_allowed: bool = True


class RegressorConfig(BaseModel):
    # pretrained: Optional[str]
    layer_dims: List[int]
    activations: List[ActivationType]
    layer_norm: List[bool]
    batch_norm: List[bool]
    dropout: float

    class Config:
        arbitrary_types_allowed: bool = True


class GraphEncoderConfig(BaseModel):
    layers: List[LayerType]
    layer_dims: List[int]
    layer_norm: List[bool]
    batch_norm: List[bool]

    class Config:
        arbitrary_types_allowed: bool = True

    @pydantic.root_validator(pre=True)
    @classmethod
    def check_ligand_encoder_config_layer_lengths(cls, values):
        if (
            len(values["layers"])
            != len(values["layer_dims"])
            != len(values["layer_norm"])
            != len(values["batch_norm"])
        ):
            raise ModuleConfigurationError(
                message="Ligand Encoder configuration of inconsistent lengths. Ensure number of layers and number of other fields match"
            )
        return values


class OptimiserType(Enum):
    SGD = "sgd"
    ADAM = "adam"


class OptimiserConfig(BaseModel):
    type: OptimiserType
    beta1: Optional[float] = 0.9
    beta2: Optional[float] = 0.999
    epsilon: Optional[float] = 1e-8
    learning_rate: float = 1e-3
    weight_decay: Optional[float] = 0
    amsgrad: Optional[bool] = False
    momentum: Optional[float] = None
    dampening: Optional[float] = None
    nesterov: Optional[float] = None

    @pydantic.root_validator(pre=False)
    @classmethod
    def check_optimiser_arguments_provided(cls, values):
        if values["type"] == OptimiserType.SGD:
            assert "learning_rate" in values
            assert "momentum" in values
            assert "weight_decay" in values
            assert "dampening" in values
            assert "nesterov" in values
        elif values["type"] == OptimiserType.ADAM:
            assert "learning_rate" in values
            assert "beta1" in values
            assert "beta2" in values
            assert "epsilon" in values
            assert "weight_decay" in values
            assert "amsgrad" in values
        return values


class ModelConfig(BaseModel):
    type: ModelType
    loss: LossConfig
    graph_encoder: GraphEncoderConfig
    classifier: Optional[ClassifierConfig] = None
    regressor: Optional[RegressorConfig] = None
    optimiser: Optional[OptimiserConfig] = None

    class Config:
        arbitrary_types_allowed: bool = True

    @pydantic.root_validator(pre=False)
    @classmethod
    def check_loss_type_correct_for_model(cls, values):
        """Checks a valid loss function is chosen for the model type"""
        if values["type"].value == (
            ModelType.GRAPH_REGRESSION or ModelType.NODE_REGRESSION
        ):
            assert values["loss"].type == LossType.MSE
        elif values["type"] == (
            ModelType.NODE_CLASSIFIER or ModelType.GRAPH_CLASSIFIER
        ):
            assert values["loss"].type != LossType.MSE
        return values

    @pydantic.root_validator(pre=True)
    @classmethod
    def check_for_decoder_config(cls, values):
        if "regressor" not in values and "classifier" not in values:
            raise ModelConfigurationError(
                message="Model incorrectly configured. Please provide either a classifier or regressor config"
            )
        return values


# Define parsers.
def parse_model_config(config: Dict[str, Any]) -> ModelConfig:
    config = config["model"]

    return ModelConfig(
        type=ModelType(config["type"]),
        graph_encoder=_parse_graph_encoder_config(config),
        classifier=_parse_classifier_config(config),
        loss=_parse_loss_config(config),
    )


def _parse_loss_config(config: Dict[str, Any]) -> LossConfig:
    config = config["loss"]

    return LossConfig(
        type=LossType(config["type"]),
        label_smoothing=config["label_smoothing"],
    )


def _parse_graph_encoder_config(config: Dict[str, Any]) -> GraphEncoderConfig:
    config = config["graph_encoder"]

    return GraphEncoderConfig(
        layers=[LayerType(l) for l in config["layers"]],
        layer_dims=config["layer_dims"],
        layer_norm=config["layer_norm"],
        batch_norm=config["batch_norm"],
    )


def _parse_classifier_config(config: Dict[str, Any]) -> ClassifierConfig:
    config = config["classifier"]

    return ClassifierConfig(
        layer_dims=config["layer_dims"],
        activations=[ActivationType(act) for act in config["activations"]],
        layer_norm=config["layer_norm"],
        batch_norm=config["batch_norm"],
        dropout=config["dropout"],
    )


def _parse_regressor_config(config: Dict[str, Any]) -> RegressorConfig:
    config = config["regressor"]

    return RegressorConfig(
        encoder=ModelType(config["encoder"]),
        pretrained=config["pretrained"],
        n_hidden=config["n_hidden"],
        d_hidden=config["d_hidden"],
    )


def parse_optimiser_config(config: Dict[str, Any]) -> OptimiserConfig:
    config = config["optimiser"]

    return OptimiserConfig(
        type=OptimiserType(config["type"]),
        beta1=config["beta1"],
        beta2=config["beta2"],
        epsilon=config["epsilon"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        amsgrad=config["amsgrad"],
        # momentum=config["momentum"],
        # dampening=config["dampening"],
        # nesterov=config["nesterov"],
    )
