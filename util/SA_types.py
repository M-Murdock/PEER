from enum import Enum, auto

class Inference(Enum):
    BAYESIAN = "Bayesian"
    MAX_ENT = "Max Entropy"
    
class Assistance(Enum):
    pass

class Arbitration(Enum):
    LINEAR = auto()
    PROBABILISTIC = auto()