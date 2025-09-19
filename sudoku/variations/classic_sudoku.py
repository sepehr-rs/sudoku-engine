# classic_sudoku.py

from typing import List, Set
from ..base_sudoku import BaseSudoku, Pos, Board
import math


class ClassicSudoku(BaseSudoku):
    """
    Standard NxN Sudoku with sub-boxes of size box_height x box_width.
    Only rows, columns, and boxes are enforced.
    """

    def __init__(
        self,
        size: int = 9,
        board: Board = None,
        box_height: int | None = None,
        box_width: int | None = None,
    ):
        super().__init__(size=size, board=board)

        if box_height is None or box_width is None:
            root = int(math.isqrt(size))
            if root * root == size:
                box_height = root
                box_width = root
            else:
                raise ValueError(
                    f"Cannot infer box dimensions for size={size}. "
                    "Provide box_height and box_width explicitly."
                )

        assert (
            box_height * box_width == size
        ), f"Box dimensions {box_height}×{box_width} must multiply to board size {size}"

        self.box_height = box_height
        self.box_width = box_width

    def regions(self) -> List[Set[Pos]]:
        """Return all standard Sudoku regions: rows, cols, and boxes."""
        N = self.size
        regions: List[Set[Pos]] = []

        # Rows
        for r in range(N):
            regions.append({(r, c) for c in range(N)})

        # Cols
        for c in range(N):
            regions.append({(r, c) for r in range(N)})

        # Boxes
        for br in range(0, N, self.box_height):
            for bc in range(0, N, self.box_width):
                box = set()
                for dr in range(self.box_height):
                    for dc in range(self.box_width):
                        box.add((br + dr, bc + dc))
                regions.append(box)

        return regions
