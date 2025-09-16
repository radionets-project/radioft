from functools import partial
from math import pi

import cufinufft
import cupy as cp
import torch


class CupyFinufft:
    """Wraper to use Finufft Type 3d3 for radio interferometry data."""

    def __init__(
        self,
        image_size,
        fov_arcsec,
        eps=1e-12,
    ):
        """Docstring"""
        self.px_size = ((fov_arcsec / 3600) * pi / 180) / image_size
        self.px_scaling = image_size**2

        self.ft = partial(cufinufft.nufft3d3, isign=1, eps=eps)
        self.ift = partial(cufinufft.nufft3d3, isign=-1, eps=eps)

    def nufft(
        self,
        sky_values,
        l_coords,
        m_coords,
        n_coords,
        u_coords,
        v_coords,
        w_coords,
        return_torch=False,
    ):
        """Docstring"""
        # Antenna coordinates (Fourier Domain - uvw coordinates)
        source_u = cp.asarray(
            2 * pi * (u_coords.flatten() * self.px_size), dtype=cp.float64
        ) % (2 * pi)
        source_v = cp.asarray(
            2 * pi * (v_coords.flatten() * self.px_size), dtype=cp.float64
        ) % (2 * pi)
        source_w = cp.asarray(
            2 * pi * (w_coords.flatten() * self.px_size), dtype=cp.float64
        ) % (2 * pi)

        # Values at source points (Source intensities)
        c_values = cp.asarray(sky_values.flatten(), dtype=cp.complex128)

        # Target coordinates (Image domain - lmn coordinates)
        target_l = cp.asarray((l_coords / self.px_size), dtype=cp.float64)
        target_m = cp.asarray((m_coords / self.px_size), dtype=cp.float64)
        target_n = cp.asarray((n_coords / self.px_size), dtype=cp.float64)

        result = self.ft(
            source_u, source_v, source_w, c_values, target_l, target_m, target_n
        )

        if return_torch:
            visibilities = torch.as_tensor(result, device="cuda")
        else:
            visibilities = result.get()

        return visibilities

    def inufft(
        self,
        visibilities,
        l_coords,
        m_coords,
        n_coords,
        u_coords,
        v_coords,
        w_coords,
        return_torch=False,
    ):
        """Docstring"""
        # Antenna coordinates (Fourier Domain - uvw coordinates)
        source_u = cp.asarray(
            2 * pi * (u_coords.flatten() * self.px_size), dtype=cp.float64
        ) % (2 * pi)
        source_v = cp.asarray(
            2 * pi * (v_coords.flatten() * self.px_size), dtype=cp.float64
        ) % (2 * pi)
        source_w = cp.asarray(
            2 * pi * (w_coords.flatten() * self.px_size), dtype=cp.float64
        ) % (2 * pi)

        # Values at source points (Source intensities)
        c_values = cp.asarray(visibilities.flatten(), dtype=cp.complex128)

        # Target coordinates (Image domain - lmn coordinates)
        target_l = cp.asarray((l_coords / self.px_size), dtype=cp.float64)
        target_m = cp.asarray((m_coords / self.px_size), dtype=cp.float64)
        target_n = cp.asarray((n_coords / self.px_size), dtype=cp.float64)

        result = (
            self.ft(
                source_u, source_v, source_w, c_values, target_l, target_m, target_n
            )
            / self.px_scaling
        )

        if return_torch:
            sky_intensities = torch.as_tensor(result, device="cuda")
        else:
            sky_intensities = result.get()

        return sky_intensities
