from typing import TypeVar

import numpy as np

State = TypeVar("State", np.ndarray, tuple)
Action = TypeVar("Action", np.ndarray, tuple)
