import numpy as np
import xarray as xr
import zarr
import matplotlib.pyplot as plt

root = zarr.open_group("data/splines_train.zarr", mode="r")

for name in root.array_keys():
    arr = root[name]
    print(name, arr.shape, arr.dtype, arr.chunks)

# Read one window and one interval: 6 coefficients over a 211x211 grid
first_interval = root["spline_coef"][0, 0]

plt.Figure()
plt.imshow(first_interval[0])
plt.show()

print("interval shape:", first_interval.shape)
print("a0 stats:", np.nanmin(first_interval[0]), np.nanmax(first_interval[0]))
print("NaNs:", np.isnan(first_interval[0]).sum())


print(root["spline_coef"][0,0,:,0,0])

for i in range(15):
    print(root["t_start"][i])

# Compare against the matching training SSH frames
ssh_train = xr.open_zarr("data/nemo_ocean.zarr")["ssh"].sel(
    time=slice("2020-06-01", "2022-12-31")
)
source_start = ssh_train.isel(time=0).values
source_next = ssh_train.isel(time=1).values

a0 = root["spline_coef"][0, 0, 0]
reconstructed_next = root["spline_coef"][0, 0].sum(axis=0)  # polynomial at tau=1

print("stored/source start times:",
      root["t_start"][0], ssh_train.time.values[0])
print("NaN masks match:", np.array_equal(np.isnan(a0), np.isnan(source_start)))
valid = np.isfinite(source_next)
print("start max error:", np.max(np.abs(a0[valid] - source_start[valid])))
print("next-hour max error:",
      np.max(np.abs(reconstructed_next[valid] - source_next[valid])))