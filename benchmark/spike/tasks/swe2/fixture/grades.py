"""Grade bookkeeping for a small class roster."""

from utils import clamp, mean, within

MAX_MARK = 100


def normalize(marks):
    """Scale marks (0 to MAX_MARK, inclusive) to fractions between 0.0 and 1.0.

    A mark outside that range is a data-entry error and raises ValueError.
    """
    for m in marks:
        if not within(m, 0, MAX_MARK):
            raise ValueError(f"mark out of range: {m}")
    return [m / MAX_MARK for m in marks]


def curve(marks, bonus):
    """Add a bonus to every mark, never going above MAX_MARK or below 0."""
    return [clamp(m + bonus, 0, MAX_MARK) for m in marks]


def class_average(marks):
    """The mean mark of the class."""
    return mean(marks)
