# sudoku-engine

A simple Python package that generates and solves m x n Sudoku puzzles. A modified version of [py-sudoku](https://github.com/jeffsieu/py-sudoku), Built by [jeffsieu](https://github.com/jeffsieu).
This library serves primarily as the backend for [Sudoku](https://flathub.org/apps/io.github.sepehr_rs.Sudoku), but can be used as a standalone Sudoku library.

## Install

```sh
# Python 3
pip3 install sudoku-engine
```
## Usage
### Classic Sudoku

```python
from sudoku import ClassicSudoku

puzzle = ClassicSudoku(size=9)
print("Puzzle:")
print(puzzle)

solution = puzzle.solve()
print("Solved:")
print(solution)
```

### Diagonal Sudoku

```python
from sudoku import DiagonalSudoku

puzzle = DiagonalSudoku(size=9)
solution = puzzle.solve()
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

```python
from sudoku.base_sudoku import PuzzleGenerator
from sudoku import ClassicSudoku

puzzle = PuzzleGenerator.make_puzzle(
    sudoku_cls=ClassicSudoku,
    size=9,
    difficulty=0.5,
    ensure_unique=True
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