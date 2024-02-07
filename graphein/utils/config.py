"""Yaml parser for config objects"""

from functools import partial

from deepdiff.operator import BaseOperator


def partial_functions_equal(func1: partial, func2: partial) -> bool:
    """
    Determine whether two partial functions are equal.

    :param func1: Partial function to check
    :type func1: partial
    :param func2: Partial function to check
    :type func2: partial
    :return: Whether the two functions are equal
    :rtype: bool
    """
    return (
        all(
            getattr(func1, attr) == getattr(func2, attr)
            for attr in ["func", "args", "keywords"]
        )
        if (isinstance(func1, partial) and isinstance(func2, partial))
        else False
    )


class PartialMatchOperator(BaseOperator):
    """Custom operator for deepdiff comparison. This operator compares whether the two partials are equal."""

    def give_up_diffing(self, level, diff_instance):
        return partial_functions_equal(level.t1, level.t2)


class PathMatchOperator(BaseOperator):
    """Custom operator for deepdiff comparison. This operator compares whether the two pathlib Paths are equal."""

    def give_up_diffing(self, level, diff_instance):
        return level.t1 == level.t2
