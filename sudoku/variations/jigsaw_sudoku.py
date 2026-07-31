# jigsaw_sudoku.py

from typing import List, Set
from ..base_sudoku import BaseSudoku, Pos, Board


class JigsawSudoku(BaseSudoku):
    def __init__(
        self,
        size: int = 9,
        board: Board = None,
        jigsaw_regions: List[Set[Pos]] | None = None,
    ):
        super().__init__(size=size, board=board)
        self.size = size
        if jigsaw_regions:
            self.jigsaw_regions = jigsaw_regions
        else:
            raise ValueError("jigsaw_regions must be provided.")

    def regions(self) -> List[Set[Pos]]:
        N = self.size
        regions: List[Set[Pos]] = []

        # Rows
        for r in range(N):
            regions.append({(r, c) for c in range(N)})

        # Cols
        for c in range(N):
            regions.append({(r, c) for r in range(N)})

        # Jigsaw Boxes
        regions.extend(self.jigsaw_regions)

        return regions
