# sudoku/utils/cage_generator.py

import random
from typing import List, Optional, Set, Tuple

from ..variations.killer_sudoku import Cage
from ..base_sudoku import Board, Pos

MOVES = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


class CageGenerator:
    """
    Partitions a *solved* NxN Sudoku board into Killer Sudoku cages.

    Each cage is a connected set of 1..max_cage_size cells whose digits
    (taken from the solved board) are all distinct. Cage sizes are chosen
    randomly within [min_cage_size, max_cage_size], shrinking as needed
    near the edges of the board or when growth gets stuck.

    Usage:
        solved = Solver(ClassicSudoku(size=9)).solve_one()
        gen = CageGenerator(size=9, seed=42)
        cages = gen.generate(solved)
    """

    def __init__(
        self,
        size: int = 9,
        min_cage_size: int = 2,
        max_cage_size: int = 4,
        seed: Optional[int] = None,
    ) -> None:
        if min_cage_size < 1:
            raise ValueError("min_cage_size must be at least 1")
        if max_cage_size < min_cage_size:
            raise ValueError("max_cage_size must be >= min_cage_size")
        if max_cage_size > size:
            raise ValueError(
                "max_cage_size cannot exceed board size (digit range is 1..size)"
            )

        self.size = size
        self.min_cage_size = min_cage_size
        self.max_cage_size = max_cage_size
        if seed is not None:
            random.seed(seed)

        self._all_cells: List[Pos] = [(r, c) for r in range(size) for c in range(size)]

    def generate(self, solved_board: Board) -> List[Cage]:
        """
        Return a list of cages that partition the board, retrying
        internally until a valid partition is found.
        """
        self._validate_board(solved_board)
        while True:
            result = self._try_generate(solved_board)
            if result is not None:
                return result

    def _validate_board(self, board: Board) -> None:
        if len(board) != self.size or any(len(row) != self.size for row in board):
            raise ValueError("solved_board dimensions do not match generator size")
        for row in board:
            if any(v is None for v in row):
                raise ValueError("solved_board must be fully filled in")

    def _try_generate(self, board: Board) -> Optional[List[Cage]]:
        claimed: Set[Pos] = set()
        cages: List[Cage] = []

        while len(claimed) < len(self._all_cells):
            start = self._random_unclaimed(claimed)
            remaining = len(self._all_cells) - len(claimed)
            target_size = min(
                random.randint(self.min_cage_size, self.max_cage_size), remaining
            )

            region = self._build_region(start, claimed, board, target_size)
            if region is None:
                # Partial failure — restart entirely
                return None

            total = sum(board[r][c] for r, c in region)
            cages.append(Cage(cells=frozenset(region), target=total))

        return cages

    def _random_unclaimed(self, claimed: Set[Pos]) -> Pos:
        return random.choice([c for c in self._all_cells if c not in claimed])

    def _neighbor(
        self,
        cell: Pos,
        move: str,
        claimed: Set[Pos],
        used_digits: Set[int],
        board: Board,
    ) -> Optional[Pos]:
        r, c = cell
        dr, dc = MOVES[move]
        nr, nc = r + dr, c + dc

        if not (0 <= nr < self.size and 0 <= nc < self.size):
            return None
        if (nr, nc) in claimed:
            return None
        if board[nr][nc] in used_digits:
            return None

        return (nr, nc)

    def _build_region(
        self, start: Pos, claimed: Set[Pos], board: Board, target_size: int
    ) -> Optional[Set[Pos]]:
        """
        Grow a single connected, digit-distinct region of up to
        `target_size` cells via DFS, starting from `start`. Mutates
        `claimed` in place on success. Accepts a smaller-than-target
        region (down to 1 cell) if growth gets stuck, rather than
        failing outright — cages don't all need to be the same size.
        """
        region: Set[Pos] = set()
        used_digits: Set[int] = set()

        stack = [(start, random.sample(list(MOVES), 4))]

        while stack:
            cell, moves = stack[-1]

            if cell not in claimed:
                claimed.add(cell)
                region.add(cell)
                used_digits.add(board[cell[0]][cell[1]])
                if len(region) == target_size:
                    return region

            if not moves:
                # Dead end at this cell — but keep whatever we've
                # already committed rather than unwinding everything;
                # a smaller connected cage is fine.
                stack.pop()
                continue

            move = moves.pop()
            nxt = self._neighbor(cell, move, claimed, used_digits, board)
            if nxt is not None:
                stack.append((nxt, random.sample(list(MOVES), 4)))

        # Stack exhausted before hitting target_size — accept whatever
        # connected, digit-distinct region we managed to build.
        return region if region else None
