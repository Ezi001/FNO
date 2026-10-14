# Neccessary imports
import time
from pathlib import Path

import numpy as np
import xarray as xr
import zarr
import matplotlib.pyplot as plt

from scripts.splines_torch import quintic_spline_batch

# Where the data is located
DATA_DIR = "/cluster/work/projects/nn8104k/esther_thesis/FNO-Nordic-Sea/data/nordic/

# trajectory length is 2 days
trajectory_length = 49
stride = 49

ds_ocean = xr.open_zarr(f"{DATA_DIR}/nemo_ocean.zarr")
#bathy = np.load(f"{DATA_DIR}/nemo_bathymetry.npz")


#print(bathy.data_vars)



# ssh = ds_ocean["ssh"]

# # Broadcast the spatial mask over time
# land_mask, ssh_broadcast = xr.broadcast(~ocean_mask, ssh)

# # Count NaNs over land and ocean
# nan_land = (ssh_broadcast.isnull() & land_mask).sum().compute().item()
# nan_ocean = (ssh_broadcast.isnull() & ~land_mask).sum().compute().item()

# print("NaNs over land:", nan_land)
# print("NaNs over ocean:", nan_ocean)


print(ds_ocean["time"].values)
#ds_forcing = xr.open_zarr(f"{DATA_DIR}/nemo_forcing.zarr")


# Training from 2020/06/01 to 2022/12/31
# Test from 2023/01/01 to 2024/12/31
ssh_train = ds_ocean["ssh"].sel(
    time=slice(
        "2020-06-01",
        "2021-12-31",
    )
)

ssh_eval = ds_ocean["ssh"].sel(
    time=slice(
        "2022-01-01",
        "2022-12-31",
    )
)

ssh_test = ds_ocean["ssh"].sel(
    time=slice(
        "2023-01-01",
        "2023-12-31",
    )
)


mean = ssh_train.mean().compute().item()
std = ssh_train.std().compute().item()

print(f"Mean: {mean}")
print(f"Std:  {std}")

ssh_train_normalised = (ssh_train - mean)/std


print(f"Mean: {ssh_train_normalised.mean().compute().item()}")
print(f"Std:  {ssh_train_normalised.std().compute().item()}")

ssh_eval_normalised = (ssh_eval - mean)/std
ssh_test_normalised = (ssh_test - mean)/std

print(f"Mean: {ssh_eval_normalised.mean().compute().item()}")
print(f"Std:  {ssh_eval_normalised.std().compute().item()}")
print(f"Mean: {ssh_test_normalised.mean().compute().item()}")
print(f"Std:  {ssh_test_normalised.std().compute().item()}")


print(ds_ocean["ssh"].isnull().sum(dim=("time")).compute())

ssh_train_filled = ssh_train_normalised.fillna(0)
ssh_eval_filled = ssh_eval_normalised.fillna(0)
ssh_test_filled = ssh_test_normalised.fillna(0)


output_path = "data/nemo_ocean_filled.zarr"


# ssh_train_filled.to_dataset(name="ssh").to_zarr(
#     output_path,
#     group="train",
#     chunk=40,
#     mode="w"
# )

# ssh_eval_filled.to_dataset(name="ssh").to_zarr(
#     output_path,
#     group="eval",
#     chunk=40,
#     mode="a"
# )

# ssh_test_filled.to_dataset(name="ssh").to_zarr(
#     output_path,
#     group="test",
#     chunk=40,
#     mode="a"
# )


from .scripts.calculate_splines import calculate_splines

# ---------------------------------------------------------
# Calculate splines
# ---------------------------------------------------------
start_time = time.time()

calculate_splines(
    ssh_train_filled,
    trajectory_length=trajectory_length,
    stride=stride,
    output_path=f"{DATA_DIR}/data_normalised/splines_train_state.zarr",
)

calculate_splines(
    ssh_eval_filled,
    trajectory_length=trajectory_length,
    stride=stride,
    output_path=f"{DATA_DIR}/data_normalised/splines_eval_state.zarr",
)


calculate_splines(
    ssh_test_filled,
    trajectory_length=trajectory_length,
    stride=stride,
    output_path=f"{DATA_DIR}/data_normalised/splines_test_state.zarr",
)

end_time = time.time()

print(f"Time taken to calculate splines: {(end_time - start_time)/60} minutes")


plt.imshow(ssh_train_filled[0])



# Count all non-NaN values
ds_ocean["ssh"].notnull().sum().compute()



plt.imshow(ssh_train[0])
plt.show()

print(ssh_train.ndim)

print(ds_ocean["missing_mask"])

# forcing_train = ds_forcing.sel(
#     time=slice(
#         "2020-06-01",
#         "2022-12-31",
#     )
# )

# print(forcing_train.ndim)
