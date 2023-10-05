# Josh Millar: edsml-jm4622

from __future__ import annotations

import os
import glob
import pyhdf
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pyhdf.SD import SD, SDC
from skimage.measure import block_reduce
from pyresample.kd_tree import resample_nearest
from pyresample.geometry import GridDefinition, SwathDefinition

from typing import Tuple

if __package__ is None:
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils import replace_nans, crop_center
else:
    from ..utils import replace_nans, crop_center


def process_file(file_name: str, df_name: str, hr_size: Tuple[int, int], ds_factor: int,) -> Tuple[np.ndarray, np.ndarray] | int:
    """
    Reads and processes MODIS data in HDF file format.

    Args:
        file_name (str): Path to the HDF file.
        df_name (str): Name of the datafield to extract.
        hr_size (int): Dimensions of HR target.
        ds_factor (int): Downsampling (i.e. coarsening) factor

    Returns:
        Tuple[np.ndarray, np.ndarray] | int: Processed data tuple containing (LR, HR) if successful, otherwise 0.

    Raises:
        pyhdf.error.HDF4Error: df_name does not exist in supplied HDF file.
    """
    try:
        hdf = SD(file_name, SDC.READ)
        data2D = hdf.select(df_name)
    except pyhdf.error.HDF4Error:
        print("Error: failed to read file - skipping")
        return 0
    data = data2D[:, :].astype(np.double)
    lat = hdf.select('Latitude')
    latitude = lat[:, :]
    lon = hdf.select('Longitude')
    longitude = lon[:, :]
    attrs = data2D.attributes(full=1)
    aoa = attrs["add_offset"]
    add_offset = aoa[0]
    fva = attrs["_FillValue"]
    _FillValue = fva[0]
    sfa = attrs["scale_factor"]
    scale_factor = sfa[0]
    data[data == _FillValue] = np.nan
    data = (data - add_offset) * scale_factor
    hdf.end()
    swath_def = SwathDefinition(lons=longitude, lats=latitude)
    min_lon, max_lon = np.min(longitude), np.max(longitude)
    min_lat, max_lat = np.min(latitude), np.max(latitude)
    x0, xinc, y0, yinc = (min_lon, 0.1, max_lat, -0.1)
    nx = int(np.floor((max_lon - min_lon) / 0.1))
    ny = int(np.floor((max_lat - min_lat) / 0.1))
    x = np.linspace(x0, x0 + xinc*nx, nx)
    y = np.linspace(y0, y0 + yinc*ny, ny)
    lon_g, lat_g = np.meshgrid(x, y)
    grid_def = GridDefinition(lons=lon_g, lats=lat_g)
    data = resample_nearest(swath_def, data, grid_def, radius_of_influence=10000, epsilon=0.5, fill_value=np.nan)
    data_m = crop_center(data, hr_size[0], hr_size[1])
    if not np.isnan(data_m).sum() == data_m.shape[0] * data_m.shape[1]:
        data_c = block_reduce(data_m, block_size=(ds_factor, ds_factor), func=np.nanmean)
        if not np.isnan(data_c).sum() > (data_c.shape[0]*data_c.shape[1]*0.25):
            if np.isnan(data_c).sum() > 0:
                nans, x = replace_nans(data_c)
                data_c[nans] = np.interp(x(nans), x(~nans), data_c[~nans])
            return data_c, data_m
    return 0


def add_arguments():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="/neodc/modis/data/MOD04_L2/collection61/*/*/*/", help="path to MODIS .hdf files")
    parser.add_argument("--df_name", default="AOD_550_Dark_Target_Deep_Blue_Combined", help="name of AOD datafield in files")
    parser.add_argument("--hr_size", default=(160, 160), type=int, help="dimensions of HR target")
    parser.add_argument("--ds_factor", default=10, type=int, help="downsampling (i.e. coarsening) factor")
    parser.add_argument("--save_to", default="/gws/nopw/j04/aopp/josh/data/aod/", help="directory to save files to")
    parser.add_argument("--train_split", default=0.80, type=int, help="proportion of dataset to include in train split")
    parser.add_argument("--val_split", default=0.75, type=int, help="proportion of dataset - train split to include in val split")
    return parser.parse_args()


if __name__ == "__main__":
    args = add_arguments()
    files = list(glob.glob(os.path.join(args.path, "*.hdf")))
    if files:
        ins, tars = [], []
        for filename in tqdm(files):
            out = process_file(filename, args.df_name, args.hr_size, args.ds_factor)
            if not out == 0:
                ins.append(out[0])
                tars.append(out[1])
        ind_t = int(args.train_split*len(ins))
        if not args.save_to[-1] == '/':
            args.save_to += '/'
        if not os.path.exists(f"{args.save_to}train"):
            os.makedirs(f"{args.save_to}train")
        if not os.path.exists(f"{args.save_to}val"):
            os.makedirs(f"{args.save_to}val")
        if not os.path.exists(f"{args.save_to}test"):
            os.makedirs(f"{args.save_to}test")
        torch.save(ins[:ind_t], f"{args.save_to}train/input_train.pth")
        torch.save(tars[:ind_t], f"{args.save_to}train/target_train.pth")
        ind_v = int(args.val_split*len(ins)-ind_t)
        torch.save(ins[ind_t:ind_v], f"{args.save_to}val/input_val.pth")
        torch.save(tars[ind_t:ind_v], f"{args.save_to}val/target_val.pth")
        torch.save(ins[ind_v:], f"{args.save_to}test/input_test.pth")
        torch.save(tars[ind_v:], f"{args.save_to}test/target_test.pth")
    else:
        print("Error: .hdf data not found at specified path")
