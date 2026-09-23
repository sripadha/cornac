import pytest

from grades import class_average, curve, normalize


def test_normalize_scales_to_unit_interval():
    assert normalize([0, 50, 100]) == [0.0, 0.5, 1.0]


def test_normalize_rejects_out_of_range():
    with pytest.raises(ValueError):
        normalize([50, 101])


def test_curve_caps_at_max():
    assert curve([95, 40], 10) == [100, 50]


def test_class_average():
    assert class_average([80, 90, 100]) == 90
