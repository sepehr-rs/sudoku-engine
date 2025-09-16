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
from sudoku.variations.classic_sudoku import ClassicSudoku

puzzle = ClassicSudoku(size=9)
print("Puzzle:")
print(puzzle)

solution = puzzle.solve()
print("Solved:")
print(solution)
```

### Diagonal Sudoku

```python
from sudoku.variations.diagonal_sudoku import DiagonalSudoku

puzzle = DiagonalSudoku(size=9)
solution = puzzle.solve()
```

### Generating Puzzles

```python
from sudoku.base_sudoku import PuzzleGenerator
from sudoku.variations.classic_sudoku import ClassicSudoku

puzzle = PuzzleGenerator.make_puzzle(
    sudoku_cls=ClassicSudoku,
    size=9,
    difficulty=0.5,
    ensure_unique=True
)
print(puzzle)
```

If you wish to raise an `UnsolvableSudoku` error when the board is invalid pass a `raising=True` parameter:

```py
puzzle.solve(raising=True)
```
