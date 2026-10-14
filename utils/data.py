"""PyTorch dataset for precomputed Nordic Seas spline trajectories."""

from __future__ import annotations

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset


class ZarrSplineDataset(Dataset):
    def __init__(self, zarr_path, forcing_path=None, trajectory_indices=None):  
        print(zarr_path)
        self.root = zarr.open(zarr_path, mode="r")

        self.spline_coef = self.root["spline_coef"]
        self.spline_end = self.root["spline_end"]
        self.spline_start = self.root["spline_start"]
        self.t_end = self.root["t_end"]
        self.t_start = self.root["t_start"]

        # self.spline_start = self._get_array(
        #     "spline_start", "starts"
        # )
        # self.spline_end = self._get_array(
        #     "spline_end", "t_end"
        # )

        self.n_trajectories_total = self.spline_coef.shape[0]
        self.n_intervals = self.spline_coef.shape[1]
        self.trajectory_length = self.n_intervals + 1
        self.time_scale = float(self.n_intervals)

        # Select which trajectories this Dataset represents
        if trajectory_indices is None:
            self.trajectory_indices = np.arange(
                self.n_trajectories_total
            )
        else:
            self.trajectory_indices = np.asarray(
                trajectory_indices,
                dtype=np.int64,
            )

            if np.any(self.trajectory_indices < 0):
                raise ValueError("Trajectory indices cannot be negative.")

            if np.any(
                self.trajectory_indices >= self.n_trajectories_total
            ):
                raise ValueError(
                    "Trajectory index exceeds number of trajectories."
                )

        self.n_trajectories = len(self.trajectory_indices)

    #     self.forcing = None

    #     if forcing_path is not None:
    #         self.forcing = np.load(
    #             forcing_path,
    #             mmap_mode="r",
    #         )

    #         if self.forcing.ndim != 4:
    #             raise ValueError(
    #                 "Forcing must have shape "
    #                 "(trajectories, height, width, channels), "
    #                 f"got {self.forcing.shape}."
    #             )

    #         if (
    #             self.forcing.shape[0]
    #             != self.n_trajectories_total
    #         ):
    #             raise ValueError(
    #                 "Forcing and spline stores must contain the "
    #                 "same number of trajectories: "
    #                 f"{self.forcing.shape[0]} vs "
    #                 f"{self.n_trajectories_total}."
    #             )

    # def _get_array(self, *names):
    #     for name in names:
    #         if name in self.root:
    #             return self.root[name]

    #     raise KeyError(
    #         f"None of {names!r} found in spline store. "
    #         f"Available arrays: {list(self.root.array_keys())}."
    #     )

    def __len__(self):
        """
        The number of trajectories 462
        """
        return len(self.spline_coef)

    def __getitem__(self, idx):
        """
        
        """
        coeffs = torch.from_numpy(self.spline_coef[idx]).float()
        starts = torch.from_numpy(self.spline_start[idx]).float()
        ends = torch.from_numpy(self.spline_end[idx]).float()

        return coeffs, starts, ends