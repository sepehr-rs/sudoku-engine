# diagonal_sudoku.py

from typing import List, Set
from .base_sudoku import BaseSudoku, Pos, Board
from .classic_sudoku import ClassicSudoku


class DiagonalSudoku(ClassicSudoku):
    """
    Standard NxN Sudoku with sub-boxes of size box_height x box_width.
    Only rows, columns, and boxes are enforced.
    """

    def regions(self) -> List[Set[Pos]]:
        """Return all standard Sudoku regions plus the two diagonals."""
        regions = super().regions()  # rows, cols, boxes
        N = self.size

        # Main diagonal (top-left to bottom-right)
        diag1 = {(i, i) for i in range(N)}
        regions.append(diag1)

        # Anti-diagonal (top-right to bottom-left)
        diag2 = {(i, N - 1 - i) for i in range(N)}
        regions.append(diag2)

        return regions
