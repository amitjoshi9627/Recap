from enum import Enum


class TrainingKeys(Enum):
    """
    Constants used by Training method
    """

    TRAIN = "train"
    RETRAIN = "retrain"


class TestingKeys(Enum):
    """
    Constants used by Testing method
    """

    EVAL = "eval"
    ALL = "all"


class ServingKeys(Enum):
    """
    Constants used by Serving method
    """

    SERVE = "serve"
    SUCCESS = "success"
    FAILURE = "failure"


class ModelKeys(Enum):
    """
    Constants used by Model
    """

    CUDA = "cuda"
    CPU = "cpu"
