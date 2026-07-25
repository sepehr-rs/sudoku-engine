# sudoku-engine

A pure-Python library that generates and solves **m x n Sudoku puzzles**. Zero external dependencies.

A modified version of [py-sudoku](https://github.com/jeffsieu/py-sudoku) by [jeffsieu](https://github.com/jeffsieu), forked and maintained by [Sepehr Rasouli](https://github.com/sepehr-rs). Serves as the backend for [Sudoku (Flatpak)](https://flathub.org/apps/io.github.sepehr_rs.Sudoku), but is fully usable as a standalone library.

## Features

- Solve any Sudoku puzzle (classic, diagonal, or custom variants)
- Generate puzzles with configurable difficulty
- Support for non-standard board sizes (4x4, 6x6, 12x12, 16x16, etc.)
- Unique-solution enforcement when generating puzzles
- Deterministic generation via optional seed
- Extensible abstract base class for creating custom Sudoku variants
- Zero dependencies

## Installation

```sh
pip install sudoku-engine
```

Requires Python 3.8+.

## Quick Start

```python
from sudoku import ClassicSudoku

# Create a random 9x9 puzzle and solve it
puzzle = ClassicSudoku(size=9)
solution = puzzle.solve()
print(solution)
```

## Usage

### Solving a Puzzle

Pass a board (2D list) with `0` or `None` for empty cells:

```python
from sudoku import ClassicSudoku

board = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]

puzzle = ClassicSudoku(size=9, board=board)
solution = puzzle.solve()

if solution:
    print(solution)
else:
    print("No solution found")
```

Output:

```
5 3 4 6 7 8 9 1 2
6 7 2 1 9 5 3 4 8
1 9 8 3 4 2 5 6 7
8 5 9 7 6 1 4 2 3
4 2 6 8 5 3 7 9 1
7 1 3 9 2 4 8 5 6
9 6 1 5 3 7 2 8 4
2 8 7 4 1 9 6 3 5
3 4 5 2 8 6 1 7 9
```

### Diagonal Sudoku

Adds main-diagonal and anti-diagonal constraints on top of classic rules:

```python
from sudoku import DiagonalSudoku

puzzle = DiagonalSudoku(size=9)
solution = puzzle.solve()
print(solution)
```

### Board Sizes

Perfect-square sizes (4, 9, 16, 25) auto-infer square boxes. Non-square sizes (6, 8, 12) require explicit box dimensions:

```python
from sudoku import ClassicSudoku

# 4x4 with 2x2 boxes (auto-inferred)
puzzle = ClassicSudoku(size=4)
solution = puzzle.solve()
print(solution)
```

```
1 2 3 4
3 4 1 2
2 1 4 3
4 3 2 1
```

```python
# 16x16 with 4x4 boxes (auto-inferred)
puzzle = ClassicSudoku(size=16)
solution = puzzle.solve()
```

Non-square sizes can be constructed with explicit box dimensions, but note that `solve()` currently does not preserve custom box dimensions when reconstructing the result. Perfect-square sizes are recommended for full functionality.

### Generating Puzzles

Use `PuzzleGenerator.make_puzzle()` to create puzzles with a controlled number of missing cells:

```python
from sudoku.base_sudoku import PuzzleGenerator
from sudoku import ClassicSudoku

puzzle = PuzzleGenerator.make_puzzle(
    sudoku_cls=ClassicSudoku,
    size=9,
    difficulty=0.5,       # remove ~50% of cells
    ensure_unique=True,   # guarantee exactly one solution
)
print(puzzle)
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `sudoku_cls` | `Type[BaseSudoku]` | *(required)* | The Sudoku class to use (`ClassicSudoku`, `DiagonalSudoku`, or a custom subclass) |
| `size` | `int` | *(required)* | Board dimension (N for NxN) |
| `difficulty` | `float` | *(required)* | Fraction of cells to remove: `0.0` (trivial) to `1.0` (all empty). Must be strictly between 0 and 1 |
| `ensure_unique` | `bool` | `True` | When `True`, only removes cells if the resulting puzzle has exactly one solution |
| `seed` | `int \| None` | `None` | Random seed for reproducible generation |
| `seed_values` | `int` | `0` | Number of pre-filled cells to place randomly before solving and removing cells |

**Difficulty guide:**

| Difficulty | Missing cells (9x9) | Approximate feel |
|---|---|---|
| 0.2 | ~16 | Very easy |
| 0.35 | ~28 | Easy |
| 0.5 | ~41 | Medium |
| 0.6 | ~49 | Hard |
| 0.7 | ~57 | Very hard |

Note: with `ensure_unique=True`, harder difficulties may keep more cells to maintain uniqueness.

### Reproducible Generation

Use `seed` for deterministic output:

```python
puzzle1 = PuzzleGenerator.make_puzzle(
    sudoku_cls=ClassicSudoku, size=9, difficulty=0.5, seed=42
)
puzzle2 = PuzzleGenerator.make_puzzle(
    sudoku_cls=ClassicSudoku, size=9, difficulty=0.5, seed=42
)

# puzzle1 and puzzle2 will be identical
print(puzzle1)
print(puzzle2)
```

### Generating Diagonal Sudoku Puzzles

Pass `DiagonalSudoku` as the class:

```python
from sudoku.base_sudoku import PuzzleGenerator
from sudoku import DiagonalSudoku

puzzle = PuzzleGenerator.make_puzzle(
    sudoku_cls=DiagonalSudoku,
    size=9,
    difficulty=0.5,
    seed=42,
)
print(puzzle)
```

### Validating a Board

Check whether a board satisfies all constraints and is solvable:

```python
from sudoku import ClassicSudoku

board = [
    [5, 3, 4, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 5, 9, 7, 6, 1, 4, 2, 3],
    [4, 2, 6, 8, 5, 3, 7, 9, 1],
    [7, 1, 3, 9, 2, 4, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 4, 5, 2, 8, 6, 1, 7, 9],
]

puzzle = ClassicSudoku(size=9, board=board)
print(puzzle.validate())  # True
```

### Checking for Multiple Solutions

```python
from sudoku import ClassicSudoku

board = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]

puzzle = ClassicSudoku(size=9, board=board)
print(puzzle.has_multiple_solutions())  # False
```

### String Representation

Boards print as space-separated values with `.` for empty cells:

```python
from sudoku import ClassicSudoku

puzzle = ClassicSudoku(size=4, board=[
    [0, 2, 0, 0],
    [0, 0, 0, 3],
    [2, 0, 0, 0],
    [0, 0, 1, 0],
])
print(puzzle)
```

```
. 2 . .
. . . 3
2 . . .
. . 1 .
```

## API Reference

### Types

Defined in `sudoku.base_sudoku`:

```python
Cell = Optional[int]        # None = empty, or int 1..N
Board = List[List[Cell]]    # 2D grid: board[row][col]
Pos = Tuple[int, int]       # (row, column) coordinate
```

### `BaseSudoku` (abstract)

Base class for all Sudoku variants.

| Method | Returns | Description |
|---|---|---|
| `__init__(size, board=None)` | | Create empty board or load from a `Board` |
| `regions()` | `List[Set[Pos]]` | **Abstract.** Return constraint regions (sets of positions that must contain unique values) |
| `extra_constraints()` | `List[Callable[[Board], bool]]` | Optional hook for additional boolean constraints. Default: `[]` |
| `validate()` | `bool` | Check board satisfies all constraints and is solvable |
| `solve()` | `BaseSudoku \| None` | Return a new instance with the solved board, or `None` if unsolvable |
| `has_multiple_solutions()` | `bool` | `True` if more than one solution exists |
| `board_copy()` | `Board` | Deep copy of the board |

### `ClassicSudoku(BaseSudoku)`

Standard Sudoku with rows, columns, and sub-boxes.

| Method | Returns | Description |
|---|---|---|
| `__init__(size=9, board=None, box_height=None, box_width=None)` | | `box_height`/`box_width` auto-inferred for perfect-square sizes |
| `regions()` | `List[Set[Pos]]` | Rows + columns + sub-boxes |

### `DiagonalSudoku(ClassicSudoku)`

Extends `ClassicSudoku` with main-diagonal and anti-diagonal constraints.

| Method | Returns | Description |
|---|---|---|
| `regions()` | `List[Set[Pos]]` | Rows + columns + sub-boxes + both diagonals |

### `PuzzleGenerator`

| Method | Returns | Description |
|---|---|---|
| `make_puzzle(sudoku_cls, size, difficulty, ensure_unique=True, seed=None, seed_values=0)` | `BaseSudoku` | Generate a puzzle of given size and difficulty |

## How It Works

The solver uses **recursive depth-first search** with two optimizations:

1. **MRV heuristic** (Minimum Remaining Values) -- always picks the empty cell with the fewest legal candidates, pruning the search tree early.
2. **Forward checking** -- maintains candidate sets per cell and prunes values from neighbors on each placement. Backtracks immediately if any cell has zero candidates.

Constraint regions (rows, columns, boxes, diagonals, etc.) are flattened into a neighbor graph at initialization. Two cells are neighbors if they appear in the same region. This makes the solver generic: any `BaseSudoku` subclass works without modifying solver code.

## License

MIT -- see [LICENSE](LICENSE).
