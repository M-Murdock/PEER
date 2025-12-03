from enum import Enum

class Inference(Enum):
    BAYESIAN = "Bayesian"
    MAX_ENT = "Max Entropy"
    
class Assistance(Enum):
    DISTRIBUTION = "Distribution"

class Arbitration(Enum):
    LINEAR = "Linear"
    PROBABILISTIC = "Probabilistic"
    ONLY_ROBOT = "Robot Action Only"