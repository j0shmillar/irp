import os
import glob
import numpy as np
from tqdm import tqdm
from pyhdf.SD import SD, SDC
from skimage.measure import block_reduce
from pyresample.kd_tree import resample_nearest
from pyresample.geometry import GridDefinition, SwathDefinition

def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]

def rm_nans(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

DATAFIELD_NAME = 'AOD_550_Dark_Target_Deep_Blue_Combined'
save_path = "/gws/nopw/j04/aopp/josh/aod/train"

i = 0

def create_files(FILE_NAME):
    global i
    hdf = SD(FILE_NAME, SDC.READ)
    data2D = hdf.select(DATAFIELD_NAME)
    data = data2D[:, :].astype(np.double)
    lat = hdf.select('Latitude')
    latitude = lat[:,:]
    lon = hdf.select('Longitude')
    longitude = lon[:,:]
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
    cellSize = 0.1
    min_lon, max_lon = np.min(longitude), np.max(longitude)
    min_lat, max_lat = np.min(latitude), np.max(latitude)
    x0, xinc, y0, yinc = (min_lon, cellSize, max_lat, -cellSize)
    nx = int(np.floor((max_lon - min_lon) / cellSize))
    ny = int(np.floor((max_lat - min_lat) / cellSize))
    x = np.linspace(x0, x0 + xinc*nx, nx)
    y = np.linspace(y0, y0 + yinc*ny, ny)
    lon_g, lat_g = np.meshgrid(x, y)
    grid_def = GridDefinition(lons=lon_g, lats=lat_g)
    data = resample_nearest(swath_def, data, grid_def,
                              radius_of_influence=10000, epsilon=0.5,
                              fill_value=np.nan)
    data_m = crop_center(data,160,160)
    if not np.isnan(data_m).sum() == data_m.shape[0] * data_m.shape[1]:
        data_c = block_reduce(data_m, block_size=(10, 10), func=np.nanmean)  # low res
        if not np.isnan(data_c).sum() > (data_c.shape[0]*data_c.shape[1]*0.25):
            if np.isnan(data_c).sum() > 0:
                nans, x = rm_nans(data_c)
                data_c[nans] = np.interp(x(nans), x(~nans), data_c[~nans])
            np.save(f"{save_path}/input/input_{i}.npy", data_c)
            np.save(f"{save_path}/target/target_{i}.npy", data_m)
            i+=1

if __name__ == "__main__":
    path = f"/neodc/modis/data/MOD04_L2/collection61/*/*/*/"
    files = list(glob.glob(os.path.join(path, "*.hdf")))
    for filename in tqdm(files):
        create_files(filename)

