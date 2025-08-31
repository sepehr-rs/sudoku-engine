from __future__ import annotations

from dataclasses import dataclass, field
from random import shuffle, seed as random_seed, randrange
from typing import List, Optional, Set, Tuple, Dict

# --------------------------
# Exceptions
# --------------------------


class UnsolvableSudoku(Exception):
    pass


# --------------------------
# Core Sudoku
# --------------------------

Cell = Optional[int]
Board = List[List[Cell]]
Pos = Tuple[int, int]


@dataclass
class Sudoku:
    """
    Unified Sudoku (supports standard boxes of size width x height and
    optional diagonal constraints).

    Empty cells are represented by None.
    """

    width: int = 3  # box width
    height: Optional[int] = None  # box height (defaults to width)
    board: Optional[Board] = None
    difficulty_value: Optional[float] = None
    diagonal: bool = (
        False  # when True, both main diagonals must contain 1..N exactly once
    )
    seed: int = field(default_factory=lambda: randrange(2**31 - 1))

    # internal
    _empty_cell_value: Cell = field(default=None, init=False, repr=False)
    _status: float = field(
        default=0.0, init=False, repr=False
    )  # mirrors public difficulty() semantics

    def __post_init__(self) -> None:
        self.height = self.height if self.height else self.width
        self.size = self.width * self.height

        assert self.width > 0, "Width cannot be less than 1"
        assert self.height > 0, "Height cannot be less than 1"
        assert self.size > 1, "Board size cannot be 1 x 1"

        if self.board is not None:
            # normalize board (coerce invalid numbers to None)
            blank_count = 0
            norm: Board = []
            for row in self.board:
                row_out: List[Cell] = []
                for v in row:
                    if isinstance(v, int) and 1 <= v <= self.size:
                        row_out.append(v)
                    else:
                        row_out.append(self._empty_cell_value)
                        blank_count += 1
                norm.append(row_out)
            self.board = norm

            if self.difficulty_value is not None:
                self._status = self.difficulty_value
            else:
                self._status = (
                    (blank_count / (self.size * self.size))
                    if self.validate() else -2.0
                )
        else:
            # seed a trivial starting board with
            # one number per row at a different column
            random_seed(self.seed)
            positions = list(range(self.size))
            shuffle(positions)
            self.board = [
                [
                    (i + 1) if i == positions[j] else self._empty_cell_value
                    for i in range(self.size)
                ]
                for j in range(self.size)
            ]
            self._status = (
                None if self.difficulty_value is None
                else self.difficulty_value
            )

    # --------------------------
    # Public API
    # --------------------------

    def solve(self, assert_solvable: bool = False) -> "Sudoku":
        """
        Solve current puzzle. Returns a new
        Sudoku instance with difficulty 0 on success.
        If unsolvable, returns an INVALID puzzle
        with difficulty = -2 (or raises if assert_solvable=True).
        """
        if not self.validate():
            if assert_solvable:
                raise UnsolvableSudoku("Invalid puzzle")
            return Sudoku(
                self.width,
                self.height,
                self.empty(self.width, self.height).board,
                -2,
                self.diagonal,
                self.seed,
            )

        solver = _Solver(
            self.board_copy(),
            self.width,
            self.height,
            self.diagonal,
        )
        solved = solver.solve_one()
        if solved is None:
            if assert_solvable:
                raise UnsolvableSudoku("No solution found")
            return Sudoku(
                self.width,
                self.height,
                self.empty(self.width, self.height).board,
                -2,
                self.diagonal,
                self.seed,
            )
        return Sudoku(
            self.width,
            self.height,
            solved,
            0.0,
            self.diagonal,
            self.seed,
        )

    def has_multiple_solutions(self) -> bool:
        """
        Returns True if the puzzle has more than one solution.
        """
        if not self.validate():
            return False
        solver = _Solver(
            self.board_copy(),
            self.width,
            self.height,
            self.diagonal,
            max_solutions=2,
        )
        solver.solve_count()
        return solver.solutions_found >= 2

    def validate(self) -> bool:  # noqa: C901
        """
        Checks current board for
        rule violations (rows, cols, boxes, and diagonals if enabled).
        Allows empty cells (None).
        """
        N = self.size
        rows = [set() for _ in range(N)]
        cols = [set() for _ in range(N)]
        boxes = [set() for _ in range(N)]
        d0: Set[int] = set()
        d1: Set[int] = set()

        for r in range(N):
            for c in range(N):
                v = self.board[r][c]
                if v is None:
                    continue
                if v in rows[r] or v in cols[c]:
                    return False
                rows[r].add(v)
                cols[c].add(v)
                b = (r // self.height) * self.width + (c // self.width)
                if v in boxes[b]:
                    return False
                boxes[b].add(v)

        if self.diagonal:
            for i in range(N):
                v = self.board[i][i]
                if v is not None:
                    if v in d0:
                        return False
                    d0.add(v)
                v = self.board[i][N - 1 - i]
                if v is not None:
                    if v in d1:
                        return False
                    d1.add(v)

        return True

    @staticmethod
    def empty(width: int, height: int) -> "Sudoku":
        size = width * height
        board = [[None for _ in range(size)] for _ in range(size)]
        return Sudoku(width, height, board, 0.0)

    def difficulty(self, difficulty: float) -> "Sudoku":
        """
        Create a puzzle by removing cells from a
        solved grid to target a given density.
        0 < difficulty < 1  (fraction of removed cells).
        If the resulting puzzle has multiple solutions,
        difficulty is set to -3 (like your original).
        """
        assert 0.0 < difficulty < 1.0, "Difficulty must be between 0 and 1"

        solved = self.solve().board  # raises only if invalid in solve()
        N = self.size
        total = N * N
        target_remove = int(difficulty * total)

        # remove cells in random order
        random_seed(self.seed)
        indices = list(range(total))
        shuffle(indices)

        puzzle = [row[:] for row in solved]
        removed = 0
        for idx in indices:
            if removed >= target_remove:
                break
            r, c = divmod(idx, N)
            if puzzle[r][c] is None:
                continue
            puzzle[r][c] = None
            removed += 1

        # check uniqueness once (fast multi-solution check)
        candidate = Sudoku(
            self.width,
            self.height,
            puzzle,
            difficulty,
            self.diagonal,
            self.seed,
        )
        if candidate.has_multiple_solutions():
            return Sudoku(
                self.width,
                self.height,
                puzzle,
                -3.0,
                self.diagonal,
                self.seed,
            )
        return candidate

    def unique_difficulty(self, difficulty: float) -> "Sudoku":
        """
        Generate a Sudoku puzzle with exactly one solution by removing cells
        from a solved grid. Ensures uniqueness after each removal attempt.

        difficulty: 0 < difficulty < 1 (fraction of cells to attempt removing)
        """
        assert 0.0 < difficulty < 1.0, "Difficulty must be between 0 and 1"

        solved = self.solve().board  # Get a solved board
        size = self.size
        total = size * size
        target_remove = int(difficulty * total)

        # Start with a fully solved puzzle
        puzzle = [row[:] for row in solved]

        # Randomize removal order
        random_seed(self.seed)
        indices = list(range(total))
        shuffle(indices)

        removed = 0
        for idx in indices:
            if removed >= target_remove:
                break
            r, c = divmod(idx, size)
            if puzzle[r][c] is None:
                continue
            saved = puzzle[r][c]
            puzzle[r][c] = None

            # Check uniqueness using your optimized solver
            solver = _Solver(
                [row[:] for row in puzzle],
                self.width,
                self.height,
                self.diagonal,
                max_solutions=2,
            )
            solver.solve_count()
            if solver.solutions_found != 1:
                # Not unique → revert removal
                puzzle[r][c] = saved
            else:
                removed += 1

        return Sudoku(
            self.width,
            self.height,
            puzzle,
            difficulty,
            self.diagonal,
            self.seed,
        )

    def get_difficulty(self) -> float:
        return float(self._status if self._status is not None else -1.0)

    def show(self) -> None:
        """
        Prints a short status header + ascii board.
        """
        status = self.get_difficulty()
        if status == -3.0:
            print("Puzzle has multiple solutions")
        elif status == -2.0:
            print("Puzzle has no solution")
        elif status == -1.0:
            print(
                "Invalid puzzle. Please solve the puzzle"
                "(puzzle.solve()), or set a difficulty (puzzle.difficulty())"
            )
        elif status == 0.0:
            print("Puzzle is solved")
        else:
            print(
                "Puzzle has exactly one solution"
                if self.validate()
                else "Invalid puzzle"
            )
        print(self._format_board_ascii())

    def show_full(self) -> None:
        print(str(self))

    # --------------------------
    # Internals / helpers
    # --------------------------

    def board_copy(self) -> Board:
        return [row[:] for row in self.board]

    def _format_board_ascii(self) -> str:
        N = self.size
        cell_len = len(str(N))
        fmt = "{0:0" + str(cell_len) + "d}"
        lines = []
        horiz = ("+-" + "-" * (cell_len + 1) * self.width) * self.height + "+"
        for r, row in enumerate(self.board):
            if r % self.height == 0:
                lines.append(horiz)
            cells = []
            for c, v in enumerate(row):
                s = fmt.format(v) if v is not None else " " * cell_len
                cells.append(s)
            # add vertical separators by sub-boxes
            line = "| "
            for b in range(self.height):
                start = b * self.width
                chunk = " ".join(cells[start: start + self.width])
                line += chunk + " | "
            lines.append(line)
        lines.append(horiz)
        return "\n".join(lines)

    def __str__(self) -> str:
        status = self.get_difficulty()
        if status == -3.0:
            d = "INVALID PUZZLE (MULTIPLE SOLUTIONS)"
        elif status == -2.0:
            d = "INVALID PUZZLE (GIVEN PUZZLE HAS NO SOLUTION)"
        elif status == -1.0:
            d = "INVALID PUZZLE"
        elif status == 0.0:
            d = "SOLVED"
        else:
            d = f"{status:.2f}"
        return (
            f"\n---------------------------\n"
            f"{self.size}x{self.size} ({self.width}x{self.height}) "
            f'{"DIAGONAL " if self.diagonal else ""}SUDOKU PUZZLE\n'
            f"Difficulty: {d}\n"
            f"---------------------------\n"
            f"{self._format_board_ascii()}\n"
        )


# --------------------------
# Fast backtracking solver with incremental candidates (MRV + forward checking)
# --------------------------


class _Solver:
    def __init__(
        self,
        board: Board,
        width: int,
        height: int,
        diagonal: bool,
        max_solutions: int = 1,
    ) -> None:  # noqa: C901
        self.board = board
        self.width = width
        self.height = height
        self.N = width * height
        self.diagonal = diagonal
        self.max_solutions = max_solutions

        # constraint sets
        self.rows: List[Set[int]] = [set() for _ in range(self.N)]
        self.cols: List[Set[int]] = [set() for _ in range(self.N)]
        self.boxes: List[Set[int]] = [set() for _ in range(self.N)]
        self.d0: Set[int] = set()
        self.d1: Set[int] = set()

        # initialize
        for r in range(self.N):
            for c in range(self.N):
                v = self.board[r][c]
                if v is None:
                    continue
                self.rows[r].add(v)
                self.cols[c].add(v)
                self.boxes[self.box_index(r, c)].add(v)
                if self.diagonal:
                    if r == c:
                        self.d0.add(v)
                    if r + c == self.N - 1:
                        self.d1.add(v)

        # candidate sets for blanks
        self.cands: Dict[Pos, Set[int]] = {}
        all_vals = set(range(1, self.N + 1))
        for r in range(self.N):
            for c in range(self.N):
                if self.board[r][c] is None:
                    ban = self.rows[r] | self.cols[c] | self.boxes[
                        self.box_index(r, c)
                    ]
                    if self.diagonal:
                        if r == c:
                            ban |= self.d0
                        if r + c == self.N - 1:
                            ban |= self.d1
                    self.cands[(r, c)] = all_vals - ban

        self.solutions_found = 0
        self.first_solution: Optional[Board] = None

    def box_index(self, r: int, c: int) -> int:
        return (r // self.height) * self.width + (c // self.width)

    def neighbors(self, r: int, c: int) -> List[Pos]:  # noqa: C901
        """
        All cells sharing row, col, box and (optionally) the two diagonals.
        """
        nbs = set()
        for cc in range(self.N):
            if cc != c:
                nbs.add((r, cc))
        for rr in range(self.N):
            if rr != r:
                nbs.add((rr, c))

        br = (r // self.height) * self.height
        bc = (c // self.width) * self.width
        for dr in range(self.height):
            for dc in range(self.width):
                rr = br + dr
                cc = bc + dc
                if (rr, cc) != (r, c):
                    nbs.add((rr, cc))

        if self.diagonal and r == c:
            for i in range(self.N):
                if i != r:
                    nbs.add((i, i))
        if self.diagonal and r + c == self.N - 1:
            for i in range(self.N):
                j = self.N - 1 - i
                if (i, j) != (r, c):
                    nbs.add((i, j))

        return list(nbs)

    def choose_cell(self) -> Optional[Pos]:
        """
        Minimum Remaining Values (MRV) heuristic.
        """
        if not self.cands:
            return None
        # pick cell with fewest candidates
        return min(self.cands.keys(), key=lambda p: len(self.cands[p]))

    def place(self, r: int, c: int, v: int) -> List[Tuple[Pos, int]]:
        """
        Place v in (r,c). Returns a list of
        (pos, removed_value) candidate removals
        for undo.
        """
        self.board[r][c] = v
        self.rows[r].add(v)
        self.cols[c].add(v)
        self.boxes[self.box_index(r, c)].add(v)
        if self.diagonal:
            if r == c:
                self.d0.add(v)
            if r + c == self.N - 1:
                self.d1.add(v)

        # remove candidates impacted by this placement (forward checking)
        removed: List[Tuple[Pos, int]] = []
        for pos in self.neighbors(r, c):
            if pos in self.cands and v in self.cands[pos]:
                self.cands[pos].remove(v)
                removed.append((pos, v))
        # cell is no longer blank
        if (r, c) in self.cands:
            del self.cands[(r, c)]
        return removed

    def unplace(
            self,
            r: int,
            c: int,
            v: int,
            removed: List[Tuple[Pos, int]]) -> None:
        """
        Undo place().
        """
        # restore candidates
        for pos, val in removed:
            if pos not in self.cands:
                self.cands[pos] = set()
            self.cands[pos].add(val)

        # restore structures
        self.board[r][c] = None
        self.rows[r].remove(v)
        self.cols[c].remove(v)
        self.boxes[self.box_index(r, c)].remove(v)
        if self.diagonal:
            if r == c:
                self.d0.remove(v)
            if r + c == self.N - 1:
                self.d1.remove(v)
        # reintroduce the blank cell with its own candidate set
        # recompute its candidates quickly given current constraints
        all_vals = set(range(1, self.N + 1))
        ban = self.rows[r] | self.cols[c] | self.boxes[self.box_index(r, c)]
        if self.diagonal:
            if r == c:
                ban |= self.d0
            if r + c == self.N - 1:
                ban |= self.d1
        self.cands[(r, c)] = all_vals - ban

    def solve_one(self) -> Optional[Board]:
        self._dfs()
        return self.first_solution

    def solve_count(self) -> int:
        self._dfs()
        return self.solutions_found

    def _dfs(self) -> None:
        if self.solutions_found >= self.max_solutions:
            return
        pos = self.choose_cell()
        if pos is None:
            # solved
            self.solutions_found += 1
            if self.first_solution is None:
                self.first_solution = [row[:] for row in self.board]
            return

        r, c = pos
        # try values in ascending order (can randomize if desired)
        for v in sorted(self.cands[pos]):
            removed = self.place(r, c, v)
            # failure detection: any neighbor left with zero candidates?
            if all(self.cands[p] for p in self.cands):
                self._dfs()
                if self.solutions_found >= self.max_solutions:
                    self.unplace(r, c, v, removed)
                    return
            self.unplace(r, c, v, removed)


# --------------------------
# Convenience subclass for Diagonal Sudoku (preserves your old API name)
# --------------------------


class DiagonalSudoku(Sudoku):
    def __init__(
        self,
        size: int = 3,
        board: Optional[Board] = None,
        difficulty: Optional[float] = None,
        seed: int = randrange(2**31 - 1),
    ) -> None:
        super().__init__(
            width=size,
            height=size,
            board=board,
            difficulty_value=difficulty,
            diagonal=True,
            seed=seed,
        )
