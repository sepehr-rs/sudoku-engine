# sudoku/utils/jigsaw_generator.py

import random
from typing import Dict, List, Optional, Set, Tuple

Pos = Tuple[int, int]

MOVES = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


class JigsawRegionGenerator:
    """
    Generates a valid set of N jigsaw regions for an NxN Sudoku board.

    Each region is a connected set of exactly N cells, and every cell
    on the board belongs to exactly one region.

    Usage:
        gen = JigsawRegionGenerator(size=9, seed=42)
        regions = gen.generate()   # List[Set[Pos]]
    """

    def __init__(self, size: int = 9, seed: Optional[int] = None) -> None:
        self.size = size
        if seed is not None:
            random.seed(seed)

        self._all_cells: List[Pos] = [(r, c) for r in range(size) for c in range(size)]

    def generate(self) -> List[Set[Pos]]:
        """
        Return a list of N connected regions that partition the board.
        Retries internally until a valid partition is found.
        """
        while True:
            result = self._try_generate()
            if result is not None:
                return result

    def _try_generate(self) -> Optional[List[Set[Pos]]]:
        """Single attempt to partition the board; returns None on failure."""
        claimed: Set[Pos] = set()
        regions: List[Set[Pos]] = []

        while len(claimed) < len(self._all_cells):
            start = self._random_unclaimed(claimed)
            region = self._build_region(start, claimed)

            if region is None:
                # Partial failure — restart entirely
                return None

            regions.append(region)

        return regions

    def _random_unclaimed(self, claimed: Set[Pos]) -> Pos:
        return random.choice([c for c in self._all_cells if c not in claimed])

    def _neighbor(self, cell: Pos, move: str, claimed: Set[Pos]) -> Optional[Pos]:
        r, c = cell
        dr, dc = MOVES[move]
        nr, nc = r + dr, c + dc

        if 0 <= nr < self.size and 0 <= nc < self.size and (nr, nc) not in claimed:
            return (nr, nc)

        return None

    def _build_region(self, start: Pos, claimed: Set[Pos]) -> Optional[Set[Pos]]:
        """
        Grow a single connected region of exactly `size` cells via DFS,
        starting from `start`. Mutates `claimed` in place on success.
        """
        N = self.size
        region: Set[Pos] = set()

        # Each stack frame: (cell, remaining directions to try)
        stack = [(start, random.sample(list(MOVES), 4))]

        while stack:
            cell, moves = stack[-1]

            if cell not in claimed:
                claimed.add(cell)
                region.add(cell)
                if len(region) == N:
                    return region

            if not moves:
                # Dead end — backtrack
                claimed.discard(cell)
                region.discard(cell)
                stack.pop()
                continue

            move = moves.pop()
            nxt = self._neighbor(cell, move, claimed)
            if nxt is not None:
                stack.append((nxt, random.sample(list(MOVES), 4)))

        return None
