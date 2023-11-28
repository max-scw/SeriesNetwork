import random
import numpy as np

def randpm1(shape = None) -> float:
    """random number [-1, +1]"""

    return (np.random.random(shape) - 0.5) * 2