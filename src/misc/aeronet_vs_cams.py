# Josh Millar: edsml-jm4622

from __future__ import annotations

import os
import torch
import pickle
import datetime
import argparse
import numpy as np
import xarray as xr
import pandas as pd
import requests as reqs
from PIL import Image
from tqdm import tqdm

from typing import Tuple

if __package__ is None:
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils import nearest
    from models.resnet import ResNet
else:
    from ..utils import nearest
    from ..models.resnet import ResNet


import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


def read_aeronet_station(station_name: str, beg_date: list[int], end_date: list[int]) -> pd.DataFrame | None:
    """
    Reads AERONET station data from AERONET web API within specified date range.

    Args:
        station_name (str): AERONET station name.
        beg_date (list[int]): Start date [year, month, day].
        end_date (list[int]): End date [year, month, day].

    Returns:
        pd.DataFrame | None: A DataFrame containing AERONET station data if read succesfully, otherwise None.

    """
    base_url = "https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_v3"
    YEAR_INDEX = 0
    MONTH_INDEX = 1
    DAY_INDEX = 2
    payload = {
        "site": station_name,
        "year": beg_date[YEAR_INDEX],
        "month": beg_date[MONTH_INDEX],
        "day": beg_date[DAY_INDEX],
        "year2": end_date[YEAR_INDEX],
        "month2": end_date[MONTH_INDEX],
        "day2": end_date[DAY_INDEX],
        "AOD20": 1,
        "AVG": 20,
        "if_no_html": 1
        }
    response = reqs.get(base_url, params=payload)

    if "html" in response.headers['content-type']:
        print(f"{response.url} not reachable.")
        return None

    def dateparse(x): return datetime.datetime.strptime(x, '%d:%m:%Y %H:%M:%S')

    return pd.read_csv(response.url, skiprows=5, na_values=-999, parse_dates={'datetime': [1, 2]}, date_parser=dateparse, index_col=0)


def read_aeronet_stations(stations_list: list[str], beg: list[int], end: list[int]) -> Tuple[list, list, pd.Series | list]:
    """
    Reads AERONET data for a list of stations within specified date range.

    Args:
        stations_list (list[str]): List of AERONET station names.
        beg (list[int]): Start date [year, month, day].
        end (list[int]): End date [year, month, day].

    Returns:
        Tuple[list, list, pd.Series | list]: Lists of latitude, longitude, and AOD @ 550nm with retrieval time for each station.

    """
    num_sites = len(stations_list)
    site_lats, site_lons, site_data = [], [], []
    for k in range(num_sites):
        print(f"Reading data for {stations_list[k] :<17} {k + 1 :>3}/{num_sites}")
        try:
            df = read_aeronet_station(stations_list[k], [beg[0], beg[1], beg[2]], [end[0], end[1], end[2]])
            site_lats.append(df['Site_Latitude(Degrees)'][0])
            site_lons.append(df['Site_Longitude(Degrees)'][0])
            A550 = df['AOD_675nm'] * (675.0 / 550.0) ** df['440-675_Angstrom_Exponent']
            site_data.append(A550)
        except pd.errors.EmptyDataError:
            print(f"\tFailed to read data for {stations_list[k] :<17}")
            continue
    return site_lats, site_lons, site_data


def add_arguments():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--cams_path", default="data/cams/cams.nc", help="path to GCM dataset")
    parser.add_argument("--cams_fname", default="aod550", help="name of target datafield in GCM dataset")
    parser.add_argument("--sl_path", default="data/stations.txt", help="path to list of AERONET stations")
    parser.add_argument("--elev_path", default="data/elev.nc4", help="path to elevation dataset")
    parser.add_argument("--elev_fname", default="PHIS", help="name of target datafield in elevation dataset")
    parser.add_argument("--weights_path", default="src/weights/best_model.pth", help="path to model weights")
    parser.add_argument("--out_path", default="data/results/cams/", help="path to save results to")
    parser.add_argument("--scale", default=False, type=bool, help="apply scale factor")
    parser.add_argument("--beg_year", default=2009, help="start year")
    parser.add_argument("--beg_month", default=1, help="start month")
    parser.add_argument("--beg_day", default=1, help="start day")
    parser.add_argument("--end_year", default=2021, help="end year")
    parser.add_argument("--end_month", default=12, help="end month")
    parser.add_argument("--end_day", default=31, help="end day")
    parser.add_argument("--time_int", default=12, type=int, help="hours interval for temp collocation")
    parser.add_argument("--dim", default=1, type=int, help="number of out channels")
    parser.add_argument("--n_channels", default=64, type=int, help="number of channels in each ResNet layer")
    parser.add_argument("--n_residual_blocks", default=4, type=int, help="number of residual blocks in ResNet")
    return parser.parse_args()


if __name__ == "__main__":
    args = add_arguments()
    AOT_550 = xr.open_dataset(args.cams_path)[args.cams_fname]
    PHIS = xr.open_dataset(args.elev_path)[args.elev_fname]
    elevations = pd.read_csv('data/aeronet_elevations.txt')
    elevations = elevations.rename(columns={'Date_Generated=27:07:2023': 'Elevation'})
    with open(args.sl_path) as f:
        for line in f:
            stations_list = line.split(',')
    site_lats, site_lons, site_data = read_aeronet_stations(stations_list, [args.beg_year, args.beg_month, args.beg_day], [args.end_year, args.end_month, args.end_day])
    site_lons_trans = np.array(site_lons) % 360
    model = ResNet(number_channels=args.n_channels, number_residual_blocks=args.n_residual_blocks, dim=args.dim)
    model.load_state_dict(torch.load(args.weights_path, map_location=torch.device('cpu')), strict=False)
    model.eval()
    for k in tqdm(range(len(site_lats))):
        actual, lr, baselines, preds = [], [], [], []
        try:
            rng = len(site_data[k])
        except IndexError:
            break
        for j in range(rng):
            t = site_data[k].index[j]
            m_in = AOT_550.sel(time=t, method='nearest')
            lat_idx = nearest(AOT_550.latitude.values, site_lats[k])[0]
            lon_idx = nearest(AOT_550.longitude.values, site_lons_trans[k])[0]
            if not args.scale:
                scale_factor = 1
            else:
                try:
                    site_elevation = np.float32(elevations[elevations.index == stations_list[k]]["Elevation"].values.max())
                except KeyError:
                    continue
                lon_idx_s = nearest(PHIS.lon.values, site_lons[k])
                lat_idx_s = nearest(PHIS.lat.values, site_lats[k])
                cell_elevation = PHIS.values.squeeze()[lat_idx_s, lon_idx_s]
                scale_factor = (cell_elevation-site_elevation)/2100
            if abs((pd.Timestamp(m_in.time.values) - t).total_seconds()/3600) > args.time_int:
                continue
            m_in_crop = m_in.values[lat_idx-8:lat_idx+8, lon_idx-8:lon_idx+8]
            lats = AOT_550.latitude.values[lat_idx-8:lat_idx+8]
            lons = AOT_550.longitude.values[lon_idx-8:lon_idx+8]
            try:
                m_in_crop = m_in_crop.reshape(1, 1, 1, 16, 16)
            except ValueError:
                continue
            actual.append(site_data[k][j])
            m_in_crop = torch.from_numpy(m_in_crop).float()
            baseline = np.array(Image.fromarray(np.array(m_in_crop).squeeze()).resize((160, 160), Image.BICUBIC))
            pred = model(m_in_crop).detach().numpy().squeeze()
            min_lat, max_lat = np.min(lats), np.max(lats)
            min_lon, max_lon = np.min(lons), np.max(lons)
            lat_coords = np.linspace(min_lat, max_lat, num=160)
            lon_coords = np.linspace(min_lon, max_lon, num=160)
            lat_idx = nearest(lat_coords, site_lats[k])
            lon_idx = nearest(lon_coords, site_lons_trans[k])
            baselines.append(baseline[lat_idx, lon_idx]*scale_factor)
            preds.append(pred[lat_idx, lon_idx]*scale_factor)
            m_in = m_in.sel(latitude=site_lats[k], longitude=site_lons_trans[k], method='nearest')
            lr.append(m_in.values*scale_factor)
        actual = np.array(actual)
        lr = np.array(lr)
        baselines = np.array(baselines)
        preds = np.array(preds)
        if not (os.path.exists(args.out_path)):
            os.makedirs(args.out_path)
        if not args.out_path[-1] == '/':
            args.out_path += '/'
        data_sets = {'actual': actual, 'lr': lr, 'baseline': baselines, 'preds': preds}
        for name, data_set in data_sets.items():
            ex_data_set = []
            try:
                with open(f"{args.out_path}{name}.pkl", 'rb') as f:
                    ex_data_set = pickle.load(f)
            except FileNotFoundError:
                pass
            for data in data_set:
                ex_data_set.append(data)
            with open(f"{args.out_path}{name}.pkl", 'wb') as f:
                pickle.dump(ex_data_set, f)
