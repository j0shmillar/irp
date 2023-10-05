# Josh Millar: edsml-jm4622

import csv
import pickle
import numpy as np

from typing import Tuple, List, Dict, Callable


def crop_center(img: np.ndarray, cropx: int, cropy: int) -> np.ndarray:
    """
    Center crop an image (as np.array).

    Args:
        img (np.ndarray): Input image.
        cropx (int): Width of crop.
        cropy (int): Height of crop.

    Returns:
        np.ndarray: Cropped image.
    """
    y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty+cropy, startx:startx+cropx]


def nearest(arr: np.ndarray, value) -> Tuple[int, ...]:
    """
    Finds the index of the nearest value in array, including nans.
    Chooses lowest index if nearest value occurs more than once.

    Args:
        arr (np.ndarray): Input array.
        value: Value to find.

    Returns:
        Tuple[int, ...]: Index of the nearest value.
    """
    arr = np.asarray(arr)
    nan_mask = np.isnan(arr)
    arr_masked = np.ma.masked_array(arr, mask=nan_mask)
    abs_diff = np.abs(arr_masked - value)
    return np.unravel_index(abs_diff.argmin(), arr.shape)


def replace_nans(y: np.ndarray) -> Tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """
    Find and replace NaNs in array.

    Args:
        y (np.ndarray): Input array.

    Returns:
        Tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]: Tuple containing NaN mask and replacement function.
    """
    return np.isnan(y), lambda z: z.nonzero()[0]


def mean(data: np.ndarray, nan_xy: list[Tuple[int, int]]) -> np.ndarray:
    """
    Replace NaN values with the mean of non-NaN values along the same column.

    Args:
        data (np.ndarray): Input data.
        nan_xy (list[Tuple[int, int]]): List of (x, y) indices with NaN values.

    Returns:
        np.ndarray: Data with NaN values replaced by means.
    """
    for x_i, y_i in nan_xy:
        data[x_i][y_i] = np.mean(data[:, [y_i]][~np.isnan(data[:, [y_i]])])
    return data


def load_csv(file_path: str) -> np.ndarray:
    """
    Loads CSV data from specified file path.

    Args:
        file_path (str): Path to CSV file.

    Returns:
        np.ndarray: Data from CSV file as np.ndarray.
    """
    with open(file_path, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    return np.concatenate(data).astype(float)


def load_pkl(file_path: str) -> np.ndarray:
    """
    Loads pkl data from specified file path.

    Args:
        file_path (str): Path to pkl file.

    Returns:
        np.ndarray: Data from pkl file as np.ndarray.
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def mse(predicted: np.ndarray, true: np.ndarray) -> float:
    """
    Mean Squared Error (MSE), handles nan values.

    Args:
        input (np.ndarray): Predicted values.
        target (np.ndarray): Ground truth values.

    Returns:
        float: MSE.
    """
    return np.nanmean(np.square(np.subtract(predicted, true)))


def nmbe(simulations: np.ndarray, observations: np.ndarray) -> float:
    """
    Normalised Mean Bias Error (NMBE), handles nan values.

    Args:
        predicted (np.ndarray): 1D simulated data.
        observed (np.ndarray): 1D observations data.

    Returns:
        float: NMBE.
    """
    mask = ~np.isnan(observations)
    simulations = simulations[mask]
    observations = observations[mask]
    return np.sum(simulations - observations) / np.sum(observations)


def kge(simulations: np.ndarray, observations: np.ndarray) -> np.ndarray:
    """
    Kling-Gupta Efficiency (KGE), handles nan values.

    Args:
        simulations (np.ndarray): 1D simulated data
        observations (np.ndarray): 1D observation data

    Returns:
        np.ndarray: KGE
    """
    mask = ~np.isnan(observations)
    simulations = simulations[mask]
    observations = observations[mask]
    sim_mean = np.mean(simulations, axis=0)
    obs_mean = np.mean(observations)
    r_num = np.sum((simulations - sim_mean) * (observations - obs_mean), axis=0)
    r_den = np.sqrt(np.sum((observations - sim_mean) ** 2, axis=0, dtype=np.float64) * np.sum((observations - obs_mean) ** 2))
    r = r_num / r_den
    alpha = np.std(simulations, axis=0) / np.std(observations, dtype=np.float64)
    beta = (np.sum(simulations, axis=0, dtype=np.float64) / np.sum(observations, dtype=np.float64))
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return kge


def generate_all_metrics(datasets: Dict[str, np.ndarray], actual: np.ndarray) -> List[List[float]]:
    """
    Generates eval metrics for different datasets compared to actual values.

    Args:
        datasets (Dict[str, np.ndarray]): Dictionary of dataset names and their corresponding data arrays.
        actual (np.ndarray): Actual data array.

    Returns:
        List[List[float]]: List of metric results for each dataset, including KGE, MSE, and NMBE.
    """
    metrics = []
    for dataset_name, dataset in datasets.items():
        KGE = kge(actual, dataset)
        MSE = mse(actual, dataset)
        NMBE = nmbe(dataset, actual)
        metrics.append([dataset_name, KGE, MSE, NMBE])
    return metrics
