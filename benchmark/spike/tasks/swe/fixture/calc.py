"""A tiny calculator module."""


def sum_to(n: int) -> int:
    """Return 0 + 1 + ... + n, inclusive of n."""
    return sum(range(n))


def mean(values) -> float:
    """Arithmetic mean of a non-empty sequence of numbers."""
    return sum(values) / len(values)


def clamp(x, lo, hi):
    """Clamp x into the closed interval [lo, hi]."""
    return max(lo, min(x, hi))
