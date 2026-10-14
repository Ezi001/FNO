from utils.data import ZarrSplineDataset
import xarray as xr
import numpy as np

from pathlib import Path
print(Path.cwd())

ds_ocean = xr.open_zarr("data/nemo_ocean.zarr")

ds = ds_ocean["ssh"][0]

print("mean", np.nanmean(ds))
print("std", np.nanstd(ds))


train = ZarrSplineDataset("./data/splines_train.zarr")

print(train.spline_start[0, 5])
print(train.n_trajectories_total)

print(train.t_end.shape)
for i in range(11):
    print(train.t_end[i])


print(train[0])
