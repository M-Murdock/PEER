from enum import Enum

class Inference(Enum):
    BAYESIAN = "Bayesian"
    MAX_ENT = "Max Entropy"
    
class Assistance(Enum):
    pass

class Arbitration(Enum):
    LINEAR = "Linear"
    PROBABILISTIC = "Probabilistic"