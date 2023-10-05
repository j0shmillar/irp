import os
import sys
import pytest
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from src.utils import crop_center, nearest, replace_nans, mean, mse, nmbe, kge  # noqa


@pytest.fixture
def sample_arr1():
    return np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]])


@pytest.fixture
def sample_arr2():
    return np.array([[1.0, 2.0, np.nan, 4.0], [1.0, 2.0, 3.0, 4.0]])


@pytest.fixture
def sample_arr3():
    return np.array([[np.nan, 2.0], [3.0, 4.0], [5.0, np.nan], [7.0, 8.0]])


@pytest.fixture
def sample_csv_data():
    return "1.0,2.0,3.0\n4.0,,6.0\n7.0,8.0,9.0"


@pytest.mark.parametrize("fixture_name, rows, cols, expected", [
    ("sample_arr1", 2, 2, np.array([[1.0, 2.0], [4.0, np.nan]])),
    ("sample_arr2", 2, 2, np.array([[2.0, np.nan], [2.0, 3.0]])),
    ("sample_arr3", 1, 3, np.array([[4.0], [np.nan], [8.0]])),
    ("sample_arr1", 3, 3, np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]])),
    ("sample_arr2", 2, 3, np.array([[2.0, np.nan], [2.0, 3.0]])),
])
def test_crop_center(fixture_name, rows, cols, expected, request):
    input_arr = request.getfixturevalue(fixture_name)
    c_arr = crop_center(input_arr, rows, cols)
    mask = ~np.isnan(c_arr)
    assert c_arr.shape == expected.shape
    assert np.allclose(c_arr[mask], expected[mask])


@pytest.mark.parametrize("fixture_name, target_value, expected_index", [
    ("sample_arr1", 4.0, (1, 0)),
    ("sample_arr2", 7.0, (0, 3)),
    ("sample_arr3", 3.0, (1, 0)),
    ("sample_arr1", 2.5, (0, 1)),
    ("sample_arr2", 1.0, (0, 0)),
])
def test_nearest(fixture_name, target_value, expected_index, request):
    input_arr = request.getfixturevalue(fixture_name)
    idx = nearest(input_arr, target_value)
    assert idx == expected_index


@pytest.mark.parametrize("fixture_name, expected_output", [
    ("sample_arr1", np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]))
])
def test_mean(fixture_name, expected_output, request):
    input_arr = request.getfixturevalue(fixture_name)
    nan_mask, _ = replace_nans(input_arr)
    i_arr = mean(input_arr.copy(), np.argwhere(nan_mask))
    assert np.allclose(i_arr, expected_output, equal_nan=True)


@pytest.mark.parametrize("fixture_name", [
    ("sample_arr1"),
    ("sample_arr2"),
    ("sample_arr3")
])
def test_replace_nans(fixture_name, request):
    input_arr = request.getfixturevalue(fixture_name)
    nan_mask, replacement_fn = replace_nans(input_arr)
    assert np.all(nan_mask == np.isnan(input_arr))
    assert callable(replacement_fn)


@pytest.mark.parametrize("predicted, true, expected_mse", [
    (np.array([1.0, 2.0, 3.0]), np.array([1.0, np.nan, 3.0]), 0.0),
    (np.array([np.nan, 2.0, 3.0]), np.array([3.0, 3.0, 3.0]), 0.5),
])
def test_mse(predicted, true, expected_mse):
    assert np.isclose(mse(predicted, true), expected_mse, rtol=1e-6)


@pytest.mark.parametrize("simulations, observations, expected_nmbe", [
    (np.array([1.0, 2.0, 3.0]), np.array([np.nan, 3.0, 1.0]), 0.25),
    (np.array([1.0, 3.0, 1.0]), np.array([np.nan, 3.0, 1.0]), 0.0),
    (np.array([1.0, 1.0, 1.0]), np.array([np.nan, 3.0, 5.0]), -0.75)
])
def test_nmbe(simulations, observations, expected_nmbe):
    assert np.isclose(nmbe(simulations, observations), expected_nmbe, rtol=1e-6)


@pytest.mark.parametrize("simulations, observations, expected_kge", [
    (np.array([1.0, 3.0, 1.0]), np.array([np.nan, 3.0, 1.0]), 1.0),
    (np.array([1.0, 2.0, 3.0]), np.array([np.nan, 3.0, 2.0]), -1.0)
])
def test_kge(simulations, observations, expected_kge):
    assert np.allclose(kge(simulations, observations), expected_kge, rtol=1e-6)


if __name__ == '__main__':
    pytest.main()
