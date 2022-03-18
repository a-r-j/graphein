import logging

import torch.functional as F
import torch.nn as nn
from multipledispatch import dispatch

from ..model_config import ActivationConfig, ActivationType

log = logging.getLogger(__name__)


@dispatch(ActivationConfig)
def build_activation(act):
    if act.type == ActivationType.TANH:
        return nn.Tanh(*act.kwargs)
    elif act.type == ActivationType.ELU:
        return nn.ELU(*act.kwargs)
    elif act.type == ActivationType.MISH:
        return nn.MISH(*act.kwargs)


@dispatch(ActivationType)
def build_activation(act):
    if act == ActivationType.TANH:
        return nn.Tanh()
    elif act == ActivationType.ELU:
        return nn.ELU()
    elif act == ActivationType.MISH:
        return nn.MISH()
    elif act == ActivationType.RELU:
        return nn.ReLU()
    elif act == ActivationType.SIGMOID:
        return nn.Sigmoid()
    elif act == ActivationType.SOFTMAX:
        return nn.Softmax()
    elif act == ActivationType.NONE:
        return None
    else:
        message = f"Activation type {act} not supported."
        log.error(message)
        raise ValueError(message)
