from CycloneSnapshot import *


def test_absolute_zero():
    assert ABSOLUTE_ZERO == 273.15


def test_earth_radius():
    assert R_E == 6371000


def test_nm_to_m():
    assert NM_TO_M == 1852


def test_wrap_360_over():
    assert wrap_360(400) == 40


def test_wrap_360_under():
    assert wrap_360(-20) == 340
