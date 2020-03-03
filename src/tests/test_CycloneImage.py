import pytest

from CycloneImage import *


def test_data_directory():
    assert os.path.isdir(DATA_DIRECTORY)


def test_wrap_over():
    assert wrap(190) == -170


def test_wrap_under():
    assert wrap(-190) == 170


def test_nm_to_degrees():
    assert nm_to_degrees(60) == 1


def test_zero_clamp():
    assert zero_clamp(-10) == 0


from datetime import datetime, timedelta
from pandas import Timestamp

test_data_A = [(timedelta(minutes=90), 3)]


@pytest.fixture()
def interpolate_start():
    return {"ISO_TIME": Timestamp(datetime(year=2020, month=1, day=1, hour=12, minute=0, second=0)), "A": 2, "B": 5,
            "C": np.nan, "D": np.nan, "E": 0, "F": "F"}


@pytest.fixture()
def interpolate_end():
    return {"ISO_TIME": Timestamp(datetime(year=2020, month=1, day=1, hour=15, minute=0, second=0)), "A": 4,
            "B": np.nan, "C": 4, "D": np.nan, "E": "E", "F": "F"}


def test_interpolate_time(interpolate_start, interpolate_end):
    dt = timedelta(minutes=90)
    result = interpolate(interpolate_start, interpolate_end, dt)
    assert result["ISO_TIME"] == datetime(year=2020, month=1, day=1, hour=13, minute=30, second=0)


def test_interpolate_A(interpolate_start, interpolate_end):
    dt = timedelta(minutes=90)
    result = interpolate(interpolate_start, interpolate_end, dt)
    assert result["A"] == 3


def test_interpolate_B(interpolate_start, interpolate_end):
    dt = timedelta(minutes=90)
    result = interpolate(interpolate_start, interpolate_end, dt)
    assert result["B"] == 5


def test_interpolate_C(interpolate_start, interpolate_end):
    dt = timedelta(minutes=90)
    result = interpolate(interpolate_start, interpolate_end, dt)
    assert result["C"] == 4


def test_interpolate_D(interpolate_start, interpolate_end):
    dt = timedelta(minutes=90)
    result = interpolate(interpolate_start, interpolate_end, dt)
    assert "D" not in result


def test_interpolate_E(interpolate_start, interpolate_end):
    dt = timedelta(minutes=90)
    result = interpolate(interpolate_start, interpolate_end, dt)
    assert result["E"] == 0


def test_interpolate_F(interpolate_start, interpolate_end):
    dt = timedelta(minutes=90)
    result = interpolate(interpolate_start, interpolate_end, dt)
    assert result["F"] == "F"
