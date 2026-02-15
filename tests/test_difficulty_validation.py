"""Test suite for difficulty validation in PuzzleGenerator.make_puzzle"""

from typing import Any, cast

import pytest

from sudoku.base_sudoku import PuzzleGenerator
from sudoku import ClassicSudoku


class TestDifficultyInvalidType:
    """Tests for TypeError cases (invalid types that cannot be coerced to float)"""

    def test_non_numeric_string_raises_type_error(self):
        """Non-numeric string like 'easy' should raise TypeError"""
        with pytest.raises(TypeError) as exc_info:
            PuzzleGenerator.make_puzzle(
                sudoku_cls=ClassicSudoku,
                size=4,
                difficulty="easy",
                ensure_unique=False,
                seed=1,
            )
        assert "difficulty" in str(exc_info.value).lower()

    def test_boolean_true_raises_type_error(self):
        """Boolean True should raise TypeError"""
        with pytest.raises(TypeError) as exc_info:
            PuzzleGenerator.make_puzzle(
                sudoku_cls=ClassicSudoku,
                size=4,
                difficulty=True,
                ensure_unique=False,
                seed=1,
            )
        assert "difficulty" in str(exc_info.value).lower()

    def test_boolean_false_raises_type_error(self):
        """Boolean False should raise TypeError"""
        with pytest.raises(TypeError) as exc_info:
            PuzzleGenerator.make_puzzle(
                sudoku_cls=ClassicSudoku,
                size=4,
                difficulty=False,
                ensure_unique=False,
                seed=1,
            )
        assert "difficulty" in str(exc_info.value).lower()

    def test_none_raises_type_error(self):
        """None should raise TypeError"""
        with pytest.raises(TypeError) as exc_info:
            PuzzleGenerator.make_puzzle(
                sudoku_cls=ClassicSudoku,
                size=4,
                difficulty=cast(Any, None),
                ensure_unique=False,
                seed=1,
            )
        assert "difficulty" in str(exc_info.value).lower()


class TestDifficultyInvalidValue:
    """Tests for ValueError cases (valid types but invalid values)"""

    def test_nan_raises_value_error(self):
        """NaN should raise ValueError"""
        with pytest.raises(ValueError) as exc_info:
            PuzzleGenerator.make_puzzle(
                sudoku_cls=ClassicSudoku,
                size=4,
                difficulty=float("nan"),
                ensure_unique=False,
                seed=1,
            )
        assert "difficulty" in str(exc_info.value).lower()

    def test_pos_inf_raises_value_error(self):
        """Positive infinity should raise ValueError"""
        with pytest.raises(ValueError) as exc_info:
            PuzzleGenerator.make_puzzle(
                sudoku_cls=ClassicSudoku,
                size=4,
                difficulty=float("inf"),
                ensure_unique=False,
                seed=1,
            )
        assert "difficulty" in str(exc_info.value).lower()

    def test_neg_inf_raises_value_error(self):
        """Negative infinity should raise ValueError"""
        with pytest.raises(ValueError) as exc_info:
            PuzzleGenerator.make_puzzle(
                sudoku_cls=ClassicSudoku,
                size=4,
                difficulty=float("-inf"),
                ensure_unique=False,
                seed=1,
            )
        assert "difficulty" in str(exc_info.value).lower()

    def test_zero_raises_value_error(self):
        """Zero should raise ValueError (range is exclusive, 0 is invalid)"""
        with pytest.raises(ValueError) as exc_info:
            PuzzleGenerator.make_puzzle(
                sudoku_cls=ClassicSudoku,
                size=4,
                difficulty=0,
                ensure_unique=False,
                seed=1,
            )
        assert "difficulty" in str(exc_info.value).lower()

    def test_one_raises_value_error(self):
        """One should raise ValueError (range is exclusive, 1 is invalid)"""
        with pytest.raises(ValueError) as exc_info:
            PuzzleGenerator.make_puzzle(
                sudoku_cls=ClassicSudoku,
                size=4,
                difficulty=1,
                ensure_unique=False,
                seed=1,
            )
        assert "difficulty" in str(exc_info.value).lower()

    def test_zero_float_raises_value_error(self):
        """Zero float should raise ValueError"""
        with pytest.raises(ValueError) as exc_info:
            PuzzleGenerator.make_puzzle(
                sudoku_cls=ClassicSudoku,
                size=4,
                difficulty=0.0,
                ensure_unique=False,
                seed=1,
            )
        assert "difficulty" in str(exc_info.value).lower()

    def test_one_float_raises_value_error(self):
        """One float should raise ValueError"""
        with pytest.raises(ValueError) as exc_info:
            PuzzleGenerator.make_puzzle(
                sudoku_cls=ClassicSudoku,
                size=4,
                difficulty=1.0,
                ensure_unique=False,
                seed=1,
            )
        assert "difficulty" in str(exc_info.value).lower()

    def test_negative_raises_value_error(self):
        """Negative value should raise ValueError"""
        with pytest.raises(ValueError) as exc_info:
            PuzzleGenerator.make_puzzle(
                sudoku_cls=ClassicSudoku,
                size=4,
                difficulty=-0.5,
                ensure_unique=False,
                seed=1,
            )
        assert "difficulty" in str(exc_info.value).lower()

    def test_greater_than_one_raises_value_error(self):
        """Value greater than 1 should raise ValueError"""
        with pytest.raises(ValueError) as exc_info:
            PuzzleGenerator.make_puzzle(
                sudoku_cls=ClassicSudoku,
                size=4,
                difficulty=1.5,
                ensure_unique=False,
                seed=1,
            )
        assert "difficulty" in str(exc_info.value).lower()


class TestDifficultyValidInput:
    """Tests for valid inputs that should pass validation"""

    def test_float_0_5_smoke_test(self):
        """Float 0.5 with size=4 should create a valid puzzle (smoke test)"""
        puzzle = PuzzleGenerator.make_puzzle(
            sudoku_cls=ClassicSudoku,
            size=4,
            difficulty=0.5,
            ensure_unique=False,
            seed=1,
        )
        assert puzzle is not None

    def test_numeric_string_0_5(self):
        """Numeric string '0.5' should be coerced and accepted"""
        puzzle = PuzzleGenerator.make_puzzle(
            sudoku_cls=ClassicSudoku,
            size=4,
            difficulty="0.5",
            ensure_unique=False,
            seed=1,
        )
        assert puzzle is not None

    def test_integer_0_5_converted_to_float(self):
        """Integer 0.5 doesn't exist, but test that valid float conversions work"""
        # Integer 0 is invalid (out of range)
        # Integer 1 is invalid (out of range)
        # Integer 0 and 1 should be tested in invalid value tests

    def test_edge_valid_0_1(self):
        """Float 0.1 should be valid (0 < 0.1 < 1)"""
        puzzle = PuzzleGenerator.make_puzzle(
            sudoku_cls=ClassicSudoku,
            size=4,
            difficulty=0.1,
            ensure_unique=False,
            seed=1,
        )
        assert puzzle is not None

    def test_edge_valid_0_9(self):
        """Float 0.9 should be valid (0 < 0.9 < 1)"""
        puzzle = PuzzleGenerator.make_puzzle(
            sudoku_cls=ClassicSudoku,
            size=4,
            difficulty=0.9,
            ensure_unique=False,
            seed=1,
        )
        assert puzzle is not None

    def test_integer_0_2(self):
        """Integer 2 should be invalid (out of range)"""
        with pytest.raises(ValueError) as exc_info:
            PuzzleGenerator.make_puzzle(
                sudoku_cls=ClassicSudoku,
                size=4,
                difficulty=2,
                ensure_unique=False,
                seed=1,
            )
        assert "difficulty" in str(exc_info.value).lower()

    def test_integer_0_3(self):
        """Integer 3 should be invalid (out of range)"""
        with pytest.raises(ValueError) as exc_info:
            PuzzleGenerator.make_puzzle(
                sudoku_cls=ClassicSudoku,
                size=4,
                difficulty=3,
                ensure_unique=False,
                seed=1,
            )
        assert "difficulty" in str(exc_info.value).lower()
