# sudoku/utils/killer_sudoku_generator.py

from typing import Optional

from .cage_generator import CageGenerator
from ..base_sudoku import Solver
from ..variations.classic_sudoku import ClassicSudoku
from ..variations.killer_sudoku import KillerSudoku


def generate_killer_sudoku(
    size: int = 9,
    box_height: Optional[int] = None,
    box_width: Optional[int] = None,
    min_cage_size: int = 2,
    max_cage_size: int = 4,
    seed: Optional[int] = None,
) -> KillerSudoku:
    """
    Generate a Killer Sudoku puzzle: a solved grid partitioned into cages,
    returned as an empty-board KillerSudoku (cage sums are the only clues).
    """
    base = ClassicSudoku(size=size, box_height=box_height, box_width=box_width)
    solved_board = Solver(base).solve_one()
    if solved_board is None:
        raise ValueError("Could not generate a solved board")

    cage_gen = CageGenerator(
        size=size,
        min_cage_size=min_cage_size,
        max_cage_size=max_cage_size,
        seed=seed,
    )
    cages = cage_gen.generate(solved_board)

    return KillerSudoku(
        size=size,
        board=None,  # puzzle starts empty; cages are the only clues
        cages=cages,
        box_height=box_height,
        box_width=box_width,
    )
