"""Convert the prepared Zarr stores into CFO training NetCDF files."""

from __future__ import annotations

import argparse
from pathlib import Path

import xarray as xr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocean-zarr", default="data/nemo_ocean.zarr")
    parser.add_argument("--forcing-zarr", default="data/nemo_forcing.zarr")
    parser.add_argument("--start-date", default="2020-06-01")
    parser.add_argument("--end-date", default="2023-12-31")
    parser.add_argument("--output-dir", default="processed")
    parser.add_argument("--dataset-id", default="2020-06-01_2023-12-31")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ocean = xr.open_zarr(args.ocean_zarr)["ssh"].sel(
        time=slice(args.start_date, args.end_date)
    )
    forcing = xr.open_zarr(args.forcing_zarr)[["msl", "u10", "v10"]].sel(
        time=slice(args.start_date, args.end_date)
    )
    ocean, forcing = xr.align(ocean, forcing, join="inner")
    if ocean.sizes.get("time", 0) < 2:
        raise ValueError("The selected date range contains fewer than two samples.")

    state = ocean.fillna(0).expand_dims(channel=["ssh"]).transpose(
        "time", "y", "x", "channel"
    )
    forcing = forcing.fillna(0).to_array("channel").transpose(
        "time", "y", "x", "channel"
    )

    state_path = output_dir / f"state_{args.dataset_id}.nc"
    forcing_path = output_dir / f"forcing_{args.dataset_id}.nc"
    print(f"Writing {state_path} with shape {state.shape}")
    state.to_netcdf(state_path)
    print(f"Writing {forcing_path} with shape {forcing.shape}")
    forcing.to_netcdf(forcing_path)
    print(f"Prepared {state.sizes['time']} aligned samples.")


if __name__ == "__main__":
    main()
