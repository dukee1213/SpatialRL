from typing import Tuple, Optional
import numpy as np
import random
import math

class NineGrid:
    def __init__(self, grid_len: int = 3, side_len: float = 232.0) -> None:
        self.grid_len = grid_len
        self.side_len = side_len
        self.cell_size = side_len / grid_len

        # Precompute private boundary arrays for each cell index
        total_cells = grid_len * grid_len
        cols = np.arange(total_cells) % grid_len
        rows = np.arange(total_cells) // grid_len

        self._x_min = cols * self.cell_size
        self._x_max = self._x_min + self.cell_size
        self._y_min = rows * self.cell_size
        self._y_max = self._y_min + self.cell_size

    def neighbors(self, idx: int) -> np.ndarray:
        n = self.grid_len
        row, col = divmod(idx, n)

        up    = idx - n    if row > 0       else -1
        left  = idx - 1    if col > 0       else -1
        right = idx + 1    if col < n - 1   else -1
        down  = idx + n    if row < n - 1   else -1

        return np.array([up, left, right, down], dtype=int)

    def cell_range(self, idx: int) -> np.ndarray:
        return np.array([
            self._x_min[idx],
            self._x_max[idx],
            self._y_min[idx],
            self._y_max[idx]
        ], dtype=float)

    def move(
        self,
        idx: int,
        position_x: Optional[float] = None,
        position_y: Optional[float] = None,
        step_size: int = 10
    ) -> np.ndarray:
        x_min = self._x_min[idx]
        x_max = self._x_max[idx]
        y_min = self._y_min[idx]
        y_max = self._y_max[idx]

        if position_x is None or position_y is None:
            new_x = random.uniform(x_min, x_max)
            new_y = random.uniform(y_min, y_max)
            return np.array([new_x, new_y], dtype=float)

        direction = np.random.choice(["up", "down", "left", "right"])
        if direction == "up":
            new_x = position_x
            new_y = position_y + step_size
        elif direction == "down":
            new_x = position_x
            new_y = position_y - step_size
        elif direction == "left":
            new_x = position_x - step_size
            new_y = position_y
        elif direction == "right":
            new_x = position_x + step_size
            new_y = position_y
        ''' If the given position_x position_y don't match the actual indexes, Handle it here '''
        new_x = min(max(new_x, x_min), x_max)
        new_y = min(max(new_y, y_min), y_max)

        return np.array([new_x, new_y], dtype=float)
# g = NineGrid()
