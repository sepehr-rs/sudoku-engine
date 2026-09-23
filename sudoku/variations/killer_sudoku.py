# sudoku/variations/killer_sudoku.py

from dataclasses import dataclass
from typing import Callable, List, Optional, Set
from ..base_sudoku import Board, Pos
from .classic_sudoku import ClassicSudoku


@dataclass(frozen=True)
class Cage:
    """A group of cells that must contain distinct digits summing to `target`."""

    cells: frozenset[Pos]
    target: int

    def __post_init__(self):
        if not self.cells:
            raise ValueError("Cage must contain at least one cell")
        if self.target <= 0:
            raise ValueError(f"Cage target must be positive, got {self.target}")


class KillerSudoku(ClassicSudoku):
    """
    Standard NxN Sudoku with sub-boxes of size box_height x box_width,
    plus Killer Sudoku cages: groups of cells that must contain no
    repeated digits and sum to a given target.
    """

    def __init__(
        self,
        size: int = 9,
        board: Optional[Board] = None,
        cages: Optional[List[Cage]] = None,
        **kwargs,
    ) -> None:
        super().__init__(size=size, board=board, **kwargs)
        if not cages:
            raise ValueError("cages must be provided.")
        self._validate_cages(cages, size)
        self.cages = cages

    @staticmethod
    def _validate_cages(cages: List[Cage], size: int) -> None:
        seen: Set[Pos] = set()
        for cage in cages:
            for pos in cage.cells:
                r, c = pos
                if not (0 <= r < size and 0 <= c < size):
                    raise ValueError(
                        f"Cage cell {pos} is out of bounds for size {size}"
                    )
                if pos in seen:
                    raise ValueError(f"Cell {pos} appears in more than one cage")
                seen.add(pos)

            n = len(cage.cells)
            if n > size:
                raise ValueError(
                    f"Cage {cage.cells} has {n} cells, more than {size} possible distinct digits"
                )

            min_possible = sum(range(1, n + 1))
            max_possible = sum(range(size - n + 1, size + 1))
            if not (min_possible <= cage.target <= max_possible):
                raise ValueError(
                    f"Cage {cage.cells} target {cage.target} is impossible for "
                    f"{n} distinct digits from 1..{size} (valid range {min_possible}-{max_possible})"
                )

        all_cells = {(r, c) for r in range(size) for c in range(size)}
        if seen != all_cells:
            raise ValueError(
                f"Cages do not cover the whole board; missing {all_cells - seen}"
            )

    def regions(self) -> List[Set[Pos]]:
        """Rows/cols/boxes plus each cage (for no-repeat pruning)."""
        regions = super().regions()
        regions.extend(set(cage.cells) for cage in self.cages)
        return regions

    def extra_constraints(self) -> List[Callable[[Board], bool]]:
        return super().extra_constraints() + [self._check_cage_sums]

    def _check_cage_sums(self, board: Board) -> bool:
        """Partial fill must never exceed target; full fill must equal it exactly."""
        for cage in self.cages:
            values = [board[r][c] for r, c in cage.cells]
            filled = [v for v in values if v is not None]
            total = sum(filled)
            if total > cage.target:
                return False
            if len(filled) == len(values) and total != cage.target:
                return False
        return True
