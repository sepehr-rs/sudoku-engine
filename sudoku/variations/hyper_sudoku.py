# hyper_sudoku.py

from typing import List, Set
from sudoku.base_sudoku import Pos
from .classic_sudoku import ClassicSudoku


class HyperSudoku(ClassicSudoku):
    """
    Standard NxN Sudoku with sub-boxes of size box_height x box_width.
    Only rows, columns, boxes, and hyper sudoku boxes are enforced.
    """

    def regions(self) -> List[Set[Pos]]:
        """Return all standard Sudoku regions plus Hyper Sudoku boxes"""
        regions = super().regions()
        N = self.size

        # Hyper Sudoku Boxes
        for br in range(1, N, self.box_height + 1):
            for bc in range(1, N, self.box_width + 1):
                box = set()
                for dr in range(3):
                    for dc in range(3):
                        box.add((br + dr, bc + dc))
                regions.append(box)

        return regions
