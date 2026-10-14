# Necessary packages
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "processed"
OUT = ROOT / "plots"
OUT.mkdir(exist_ok=True)


state_file =  PROCESSED / "state_1980-01-01_1980-01-07.nc"
forcing_file = PROCESSED / "forcing_1980-01-01_1980-01-07.nc"
mask_file =  PROCESSED / "mask_1980-01-01_1980-01-07.nc"



# ---------------------------------------------------------
# Open datasets
# ---------------------------------------------------------

state_ds = xr.open_dataset(
    state_file,
    engine="h5netcdf",
    decode_times=False
).drop_vars("time_centered", errors="ignore")

forcing_ds = xr.open_dataset(
    forcing_file,
    engine="h5netcdf",
    decode_times=False
).drop_vars("time_centered", errors="ignore")

mask_ds = xr.open_dataset(
    mask_file,
    engine="h5netcdf",
    decode_times=False
).drop_vars("time_centered", errors="ignore")

print("Dataset opened successfully")
print("Dimensions:", state_ds.sizes)
print("Data variables:", list(state_ds.data_vars))
print("Coordinates:", list(state_ds.coords))
print("Variables:", list(state_ds.variables))

print("time dtype:", state_ds["time"].dtype)
print("time_centered dtype:", state_ds["time_centered"].dtype)

print("time values:")
print(state_ds["time"].values)

print("time_centered values:")
print(state_ds["time_centered"].values)

# print("\nFORCING:")
# print(forcing_ds)

# print("\nMASK:")
# print(mask_ds)

for name in state_ds.variables:
    var = state_ds[name]
    print(
        name,
        "dims=", var.dims,
        "dtype=", var.dtype,
        "shape=", var.shape
    )


# ---------------------------------------------------------
# Extract the single DataArray from each dataset
# ---------------------------------------------------------

def get_data_array(ds):
    if len(ds.data_vars) != 1:
        raise ValueError(
            f"Expected one data variable, found {list(ds.data_vars)}"
        )

    return ds[list(ds.data_vars)[0]]


state_da = get_data_array(state_ds)
forcing_da = get_data_array(forcing_ds)
mask_da = get_data_array(mask_ds)


# ---------------------------------------------------------
# Basic information
# ---------------------------------------------------------

print("\nSTATE TIME:", state_da.sizes.get("time"))
if "time" in state_da.coords:
    print("STATE TIMES:", state_da.time.values)

print("\nFORCING TIME:", forcing_da.sizes.get("time"))
if "time" in forcing_da.coords:
    print("FORCING TIMES:", forcing_da.time.values)

print("\nSTATE SHAPE:", state_da.shape)
print("FORCING SHAPE:", forcing_da.shape)
print("MASK SHAPE:", mask_da.shape)


# ---------------------------------------------------------
# Get first timestep
# ---------------------------------------------------------

def first_timestep(da):
    if "time" in da.dims:
        return da.isel(time=0)
    return da


state_t0 = first_timestep(state_da)
forcing_t0 = first_timestep(forcing_da)
mask_t0 = first_timestep(mask_da)

mask_t0 = mask_t0.squeeze(drop=True)


print("\nSTATE first timestep:", state_t0.dims)
print("FORCING first timestep:", forcing_t0.dims)
print("MASK:", mask_t0.dims)


# ---------------------------------------------------------
# Curvilinear coordinates
# ---------------------------------------------------------

def get_curvilinear_coords(da):

    if "nav_lon" not in da.coords:
        raise ValueError("Missing nav_lon coordinate")

    if "nav_lat" not in da.coords:
        raise ValueError("Missing nav_lat coordinate")

    lon = da["nav_lon"].values
    lat = da["nav_lat"].values

    if lon.ndim != 2 or lat.ndim != 2:
        raise ValueError(
            f"Expected 2D coordinates, got "
            f"lon={lon.ndim}D, lat={lat.ndim}D"
        )

    if lon.shape != da.shape or lat.shape != da.shape:
        raise ValueError(
            f"Coordinate shape mismatch: "
            f"field={da.shape}, "
            f"lon={lon.shape}, "
            f"lat={lat.shape}"
        )

    return lon, lat


# ---------------------------------------------------------
# Curvilinear cell edges
# ---------------------------------------------------------

def curvilinear_edges(coord):

    padded = np.empty(
        (coord.shape[0] + 2, coord.shape[1] + 2),
        dtype=coord.dtype
    )

    padded[1:-1, 1:-1] = coord

    padded[0, 1:-1] = 2 * coord[0] - coord[1]
    padded[-1, 1:-1] = 2 * coord[-1] - coord[-2]

    padded[1:-1, 0] = 2 * coord[:, 0] - coord[:, 1]
    padded[1:-1, -1] = 2 * coord[:, -1] - coord[:, -2]

    padded[0, 0] = (
        padded[0, 1] + padded[1, 0] - padded[1, 1]
    )

    padded[0, -1] = (
        padded[0, -2] + padded[1, -1] - padded[1, -2]
    )

    padded[-1, 0] = (
        padded[-2, 0] + padded[-1, 1] - padded[-2, 1]
    )

    padded[-1, -1] = (
        padded[-2, -1] + padded[-1, -2] - padded[-2, -2]
    )

    return (
        padded[:-1, :-1]
        + padded[1:, :-1]
        + padded[:-1, 1:]
        + padded[1:, 1:]
    ) / 4


# ---------------------------------------------------------
# Plot a curvilinear field
# ---------------------------------------------------------

def plot_2d(da, ax, title, cmap="viridis"):

    da = da.squeeze(drop=True)

    lon, lat = get_curvilinear_coords(da)

    lon_edges = curvilinear_edges(lon)
    lat_edges = curvilinear_edges(lat)

    mesh = ax.pcolormesh(
        lon_edges,
        lat_edges,
        da.values,
        shading="flat",
        cmap=cmap
    )

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    plt.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)

    return mesh


# =========================================================
# 1. Plot all state variables
# =========================================================

if "channel" in state_t0.dims:

    state_channels = list(state_t0.channel.values)

    fig, axes = plt.subplots(
        1,
        len(state_channels),
        figsize=(7 * len(state_channels), 6),
        squeeze=False
    )

    axes = axes.ravel()

    for i, channel in enumerate(state_channels):

        field = state_t0.sel(channel=channel)

        print(
            f"State {channel}: "
            f"shape={field.shape}, "
            f"min={field.min().item()}, "
            f"max={field.max().item()}"
        )

        plot_2d(
            field,
            axes[i],
            f"State: {channel}"
        )

else:

    fig, ax = plt.subplots(figsize=(8, 6))

    plot_2d(
        state_t0,
        ax,
        "State"
    )

    axes = [ax]


fig.suptitle("State variables — first timestep")

fig.tight_layout()

out_path = OUT / "state_overview.png"

fig.savefig(
    out_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)

print(f"Saved: {out_path}")


# =========================================================
# 2. Plot all forcing variables
# =========================================================

if "channel" in forcing_t0.dims:

    forcing_channels = list(forcing_t0.channel.values)

    fig, axes = plt.subplots(
        1,
        len(forcing_channels),
        figsize=(7 * len(forcing_channels), 6),
        squeeze=False
    )

    axes = axes.ravel()

    for i, channel in enumerate(forcing_channels):

        field = forcing_t0.sel(channel=channel)

        # Convert MSL from Pa to hPa
        if channel == "msl":
            field = field / 100.0
            title = "Forcing: MSL [hPa]"
        elif channel == "u10":
            title = "Forcing: u10 [m/s]"
        elif channel == "v10":
            title = "Forcing: v10 [m/s]"
        else:
            title = f"Forcing: {channel}"

        print(
            f"Forcing {channel}: "
            f"shape={field.shape}, "
            f"min={field.min().item()}, "
            f"max={field.max().item()}"
        )

        plot_2d(
            field,
            axes[i],
            title
        )

    fig.suptitle("Forcing variables — first timestep")

else:

    fig, ax = plt.subplots(figsize=(8, 6))

    plot_2d(
        forcing_t0,
        ax,
        "Forcing"
    )

    axes = [ax]


fig.tight_layout()

out_path = OUT / "forcing_overview.png"

fig.savefig(
    out_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)

print(f"Saved: {out_path}")


# =========================================================
# 3. Plot ocean mask
# =========================================================

fig, ax = plt.subplots(figsize=(8, 6))

plot_2d(
    mask_t0.astype(float),
    ax,
    "Ocean mask",
    cmap="gray"
)

fig.tight_layout()

out_path = OUT / "ocean_mask.png"

fig.savefig(
    out_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)

print(f"Saved: {out_path}")


# =========================================================
# 4. Plot MSL over ocean only
# =========================================================

msl = forcing_t0.sel(channel="msl") / 100.0

# Make sure mask has same coordinates
msl_ocean = msl.where(mask_t0)

fig, ax = plt.subplots(figsize=(8, 6))

plot_2d(
    msl_ocean,
    ax,
    "Mean sea-level pressure over ocean [hPa]"
)

fig.tight_layout()

out_path = OUT / "msl_ocean.png"

fig.savefig(
    out_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)

print(f"Saved: {out_path}")


# =========================================================
# Final checks
# =========================================================

print("\nCoordinate ranges:")

print(
    "State latitude:",
    float(state_da.nav_lat.min()),
    "to",
    float(state_da.nav_lat.max())
)

print(
    "State longitude:",
    float(state_da.nav_lon.min()),
    "to",
    float(state_da.nav_lon.max())
)

print(
    "Forcing latitude:",
    float(forcing_da.nav_lat.min()),
    "to",
    float(forcing_da.nav_lat.max())
)

print(
    "Forcing longitude:",
    float(forcing_da.nav_lon.min()),
    "to",
    float(forcing_da.nav_lon.max())
)

print("\nDone.")