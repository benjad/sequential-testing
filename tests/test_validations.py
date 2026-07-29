import pytest
from src.validations import validate_probability, validate_lenght, validate_options

from contextlib import nullcontext as does_not_raise
import pytest


@pytest.mark.parametrize(
    "input,expectation",
    [
        (0.2, does_not_raise()),
        (0.9, does_not_raise()),
        (3, pytest.raises(ValueError)),
        (-1, pytest.raises(ValueError)),
    ],
)
def test_validate_probability(input, expectation):
    with expectation:
        validate_probability(input)


@pytest.mark.parametrize(
    "list_a,list_b,expectation",
    [
        ([1], [2], does_not_raise()),
        ([1, 2], [2, 3], does_not_raise()),
        ([7, 1, 2], [2, 3], pytest.raises(ValueError)),
        ([7, 1, 2], [2, 3, 6, 2], pytest.raises(ValueError)),
    ],
)
def test_validate_lenght(list_a, list_b, expectation):
    with expectation:
        validate_lenght(list_a, list_b)


@pytest.mark.parametrize(
    "input,list_options,expectation",
    [
        ("a", ["c", "a", "e"], does_not_raise()),
        ("c", ["c", "a", "e"], does_not_raise()),
        ("f", ["c", "a", "e"], pytest.raises(ValueError)),
        ("j", ["c", "a", "e"], pytest.raises(ValueError)),
    ],
)
def test_validate_options(input, list_options, expectation):
    with expectation:
        validate_options(input, list_options)
