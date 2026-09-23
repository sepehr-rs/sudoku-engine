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

### Hyper Sudoku

Hyper Sudoku adds four extra 3x3 "window" regions that must also contain every digit exactly once, in addition to the standard rows, columns, and boxes.

```python
from sudoku import HyperSudoku

puzzle = HyperSudoku(size=9)
solution = puzzle.solve()
```

### Jigsaw Sudoku

Jigsaw Sudoku replaces the standard 3x3 boxes with irregularly-shaped connected regions, each still containing every digit exactly once. Regions must be supplied when constructing the puzzle:

```python
from sudoku import JigsawSudoku, JigsawRegionGenerator

jigsaw_regions = JigsawRegionGenerator(size=9).generate()
puzzle = JigsawSudoku(size=9, jigsaw_regions=jigsaw_regions)
solution = puzzle.solve()
```

### Killer Sudoku

Killer Sudoku adds cages: connected groups of cells that must contain no repeated digits and sum to a given target.

```python
from sudoku import generate_killer_sudoku

puzzle = generate_killer_sudoku(size=9)
print("Puzzle:")
print(puzzle)
for cage in puzzle.cages:
    print(cage.cells, "->", cage.target)

solution = puzzle.solve()
print("Solved:")
print(solution)
```

You can also build a `KillerSudoku` directly from your own cages:

```python
from sudoku import KillerSudoku, Cage

cages = [
    Cage(cells=frozenset({(0, 0), (0, 1)}), target=10),
    # ... cages must cover every cell on the board exactly once
]
puzzle = KillerSudoku(size=9, cages=cages)
```

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

Jigsaw regions and Killer Sudoku cages have their own dedicated generators, since both need to partition the board into connected regions rather than simply removing digits from a solved grid:

```python
from sudoku import JigsawRegionGenerator, CageGenerator

jigsaw_regions = JigsawRegionGenerator(size=9).generate()
```

For Killer Sudoku, `generate_killer_sudoku` (shown above) wraps this end-to-end — solving a grid, partitioning it into cages, and returning a ready-to-solve `KillerSudoku` puzzle. If you want the cages for a board you've already solved yourself, use `CageGenerator` directly:

```python
from sudoku.base_sudoku import Solver
from sudoku import ClassicSudoku, CageGenerator

solved = Solver(ClassicSudoku(size=9)).solve_one()
cages = CageGenerator(size=9).generate(solved)
```

If you wish to raise an `UnsolvableSudoku` error when the board is invalid pass a `raising=True` parameter:

```py
puzzle.solve(raising=True)
```
