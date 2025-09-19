# base_sudoku.py

from abc import ABC, abstractmethod
from typing import List, Set, Tuple, Optional, Callable, Type
import random

Cell = Optional[int]
Board = List[List[Cell]]
Pos = Tuple[int, int]


class BaseSudoku(ABC):
    """
    Abstract base for all Sudoku variants.
    Handles board state + solving/validation hooks,
    while subclasses supply specific region/constraint logic.
    """

    def __init__(self, size: int, board: Optional[Board] = None) -> None:
        self.size = size
        self.board: Board = (
            [[None] * size for _ in range(size)] if board is None else board
        )

    @abstractmethod
    def regions(self) -> List[Set[Pos]]:
        """Return sets of positions (each must contain 1..N)."""
        ...

    def extra_constraints(self) -> List[Callable[[Board], bool]]:
        """
        Optional extra checks (e.g., diagonals, killer cages).
        Defaults to none.
        """
        return []

    def validate(self) -> bool:
        """
        Check if board respects all constraints (rows, cols, regions, extras).
        """
        # validate rows & cols
        for i in range(self.size):
            if not self._check_unit([(i, j) for j in range(self.size)]):
                return False
            if not self._check_unit([(j, i) for j in range(self.size)]):
                return False

        # validate regions
        for region in self.regions():
            if not self._check_unit(region):
                return False

        # extra constraints
        for check in self.extra_constraints():
            if not check(self.board):
                return False

        solver = Solver(self, max_solutions=1)
        return solver.solve_one() is not None

    def solve(self) -> Optional["BaseSudoku"]:
        solver = Solver(self)
        solved = solver.solve_one()
        if solved is None:
            return None
        return self.__class__(size=self.size, board=solved)

    def has_multiple_solutions(self) -> bool:
        solver = Solver(self, max_solutions=2)
        return solver.solve_count() > 1

    def board_copy(self) -> Board:
        return [row[:] for row in self.board]

    def _check_unit(self, positions: List[Pos]) -> bool:
        """Helper: ensure no duplicate values in a given unit."""
        seen = set()
        for r, c in positions:
            v = self.board[r][c]
            if v is None:
                continue
            if v in seen:
                return False
            seen.add(v)
        return True

    def __str__(self) -> str:
        return "\n".join(" ".join(str(v or ".") for v in row) for row in self.board)


class Solver:
    """
    Generic backtracking solver.
    Uses BaseSudoku's regions() + extra_constraints() to guide search.
    """

    def __init__(self, puzzle: BaseSudoku, max_solutions: int = 1):
        self.puzzle = puzzle
        self.N = puzzle.size
        self.max_solutions = max_solutions

        # Working copy of board
        self.board = puzzle.board_copy()

        # Precompute neighbors from regions
        self.neighbors: dict[Pos, set[Pos]] = self._build_neighbors()

        # Candidate sets
        all_vals = set(range(1, self.N + 1))
        self.cands: dict[Pos, set[int]] = {
            (r, c): all_vals - self._used_in_neighbors(r, c)
            for r in range(self.N)
            for c in range(self.N)
            if self.board[r][c] is None
        }

        # Results
        self.solutions_found = 0
        self.first_solution: Optional[Board] = None

    def _build_neighbors(self) -> dict[Pos, set[Pos]]:
        """
        Precompute all neighbors of each cell from regions.
        Two cells are neighbors if they appear in the same region.
        """
        neighbors: dict[Pos, set[Pos]] = {
            (r, c): set() for r in range(self.N) for c in range(self.N)
        }
        for region in self.puzzle.regions():
            for r1, c1 in region:
                for r2, c2 in region:
                    if (r1, c1) != (r2, c2):
                        neighbors[(r1, c1)].add((r2, c2))
        return neighbors

    def _used_in_neighbors(self, r: int, c: int) -> set[int]:
        """Values already present among a cell’s neighbors."""
        vals = set()
        for rr, cc in self.neighbors[(r, c)]:
            v = self.board[rr][cc]
            if v is not None:
                vals.add(v)
        return vals

    def choose_cell(self) -> Optional[Pos]:
        """MRV heuristic: pick cell with fewest candidates."""
        if not self.cands:
            return None
        return min(self.cands, key=lambda p: len(self.cands[p]))

    def place(self, r: int, c: int, v: int) -> List[Tuple[Pos, int]]:
        """Place value and update candidate sets."""
        self.board[r][c] = v
        removed: list[tuple[Pos, int]] = []
        for nb in self.neighbors[(r, c)]:
            if nb in self.cands and v in self.cands[nb]:
                self.cands[nb].remove(v)
                removed.append((nb, v))
        if (r, c) in self.cands:
            del self.cands[(r, c)]
        return removed

    def unplace(self, r: int, c: int, v: int, removed: List[Tuple[Pos, int]]):
        """Undo place()."""
        self.board[r][c] = None
        for pos, val in removed:
            self.cands[pos].add(val)
        self.cands[(r, c)] = set(range(1, self.N + 1)) - self._used_in_neighbors(r, c)

    def _dfs(self):
        if self.solutions_found >= self.max_solutions:
            return
        pos = self.choose_cell()
        if pos is None:
            # candidate solution
            # run extra constraints before accepting
            if all(check(self.board) for check in self.puzzle.extra_constraints()):
                self.solutions_found += 1
                if self.first_solution is None:
                    self.first_solution = [row[:] for row in self.board]
            return

        r, c = pos
        for v in random.sample(sorted(self.cands[pos]), len(self.cands[pos])):
            removed = self.place(r, c, v)
            if all(self.cands[p] for p in self.cands):
                self._dfs()
                if self.solutions_found >= self.max_solutions:
                    self.unplace(r, c, v, removed)
                    return
            self.unplace(r, c, v, removed)

    def solve_one(self) -> Optional[Board]:
        self._dfs()
        return self.first_solution

    def solve_count(self) -> int:
        self._dfs()
        return self.solutions_found


class PuzzleGenerator:
    """
    Utilities for creating Sudoku puzzles from any BaseSudoku subclass.
    """

    @staticmethod
    def make_puzzle(
        sudoku_cls: Type[BaseSudoku],
        size: int,
        difficulty: float,
        ensure_unique: bool = True,
        seed: Optional[int] = None,
        seed_values: int = 0,
    ) -> BaseSudoku:  # noqa: C901
        """
        Create a puzzle of given size and difficulty.

        difficulty: 0 < difficulty < 1 (fraction of cells to remove)
        ensure_unique: enforce uniqueness of solution
        """
        assert 0 < difficulty < 1, "Difficulty must be between 0 and 1"

        full = sudoku_cls(size=size)

        if seed is not None:
            random.seed(seed)

        PuzzleGenerator._prefill_cells(full, size, seed_values)
        solver = Solver(full)
        solved_board = solver.solve_one()
        if solved_board is None:
            raise ValueError("Could not generate a solved board")

        puzzle = sudoku_cls(size=size, board=solved_board)
        PuzzleGenerator._remove_cells(puzzle, size, difficulty, ensure_unique, seed)
        return puzzle

    @staticmethod
    def _prefill_cells(full: BaseSudoku, size: int, seed_values: int):
        """Optionally prefill some cells before generating puzzle."""
        for _ in range(seed_values):
            r, c = random.randrange(size), random.randrange(size)
            if full.board[r][c] is not None:
                continue  # skip already filled
            candidates = set(range(1, size + 1))
            # remove values already in row or column
            for rr, cc in [(r, j) for j in range(size)] + [(i, c) for i in range(size)]:
                if full.board[rr][cc] in candidates:
                    candidates.remove(full.board[rr][cc])
            if candidates:
                full.board[r][c] = random.choice(list(candidates))

    @staticmethod
    def _remove_cells(
        puzzle: BaseSudoku,
        size: int,
        difficulty: float,
        ensure_unique: bool,
        seed: Optional[int] = None,
    ):
        """Remove cells from the solved puzzle to create the final puzzle."""
        total = size * size
        target_remove = int(difficulty * total)

        indices = list(range(total))
        if seed is not None:
            random.seed(seed)
        random.shuffle(indices)

        removed = 0
        for idx in indices:
            if removed >= target_remove:
                break
            r, c = divmod(idx, size)
            if puzzle.board[r][c] is None:
                continue
            saved = puzzle.board[r][c]
            puzzle.board[r][c] = None

            if ensure_unique:
                solver = Solver(puzzle, max_solutions=2)
                if solver.solve_count() != 1:
                    # revert removal if not unique
                    puzzle.board[r][c] = saved
                else:
                    removed += 1
            else:
                removed += 1
