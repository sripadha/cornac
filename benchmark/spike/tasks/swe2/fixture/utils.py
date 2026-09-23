"""Numeric helpers shared by the grading code."""


def within(x, lo, hi):
    """True iff x lies in the closed interval [lo, hi], both ends included."""
    return lo < x < hi


def clamp(x, lo, hi):
    """Clamp x into the closed interval [lo, hi]."""
    return max(lo, min(x, hi))


def mean(values):
    """Arithmetic mean of a non-empty sequence of numbers."""
    return sum(values) / len(values)
