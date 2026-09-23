from calc import clamp, mean, sum_to


def test_sum_to():
    assert sum_to(4) == 10  # 0 + 1 + 2 + 3 + 4
    assert sum_to(1) == 1


def test_mean():
    assert mean([2, 4, 6]) == 4


def test_clamp():
    assert clamp(15, 0, 10) == 10
    assert clamp(-3, 0, 10) == 0
    assert clamp(5, 0, 10) == 5
