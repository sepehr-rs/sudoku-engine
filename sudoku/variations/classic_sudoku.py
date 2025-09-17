# classic_sudoku.py

from typing import List, Set
from sudoku.base_sudoku import BaseSudoku, Pos, Board


class ClassicSudoku(BaseSudoku):
    """
    Standard NxN Sudoku with sub-boxes of size box_height x box_width.
    Only rows, columns, and boxes are enforced.
    """

    def __init__(
        self,
        size: int = 9,
        board: Board = None,
        box_height: int = 3,
        box_width: int = 3,
    ):
        super().__init__(size=size, board=board)
        assert (
            box_height * box_width == size
        ), "Box dimensions must multiply to board size"
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
