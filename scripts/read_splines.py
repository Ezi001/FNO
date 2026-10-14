import zarr
import numpy as np
import xarray as xr

ssh = xr.open_zarr("data/nemo_ocean.zarr")
original = ssh["ssh"][0]

print(f"Original shape {original}")

root = zarr.open("data/splines_train.zarr", mode="r")

print(root.tree())

coeff = root["spline_coef"]

print("shape:", coeff.shape)
print("dtype:", coeff.dtype)

sample = coeff[0]

print("sample shape:", sample.shape)
print("min:", np.nanmin(sample))
print("max:", np.nanmax(sample))
print("mean:", np.nanmean(sample))
print("NaNs:", np.isnan(sample).sum())
print("Infs:", np.isinf(sample).sum())

spline_at_0 = coeff[0, 0, 0]

error = spline_at_0 - original

print("MAE:", np.nanmean(np.abs(error)))
print("RMSE:", np.sqrt(np.nanmean(error**2)))
print("max error:", np.nanmax(np.abs(error)))