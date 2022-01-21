from typing import TypeVar

import numpy as np

State = TypeVar("State", int, np.ndarray, tuple)
Action = TypeVar("Action", int, np.ndarray, tuple)
