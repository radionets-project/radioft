import warnings
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

        self.ft = partial(cufinufft.nufft3d3, isign=-1, eps=eps)
        self.ift = partial(cufinufft.nufft3d3, isign=+1, eps=eps)

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
        """Calculate the fast Fourier transform with non-uniform source
        and non-uniform target coordinates.
        """
        # Sky coordinates (Image domain - lmn coordinates)
        source_l = cp.asarray((l_coords / self.px_size), dtype=cp.float64)
        source_m = cp.asarray((m_coords / self.px_size), dtype=cp.float64)
        source_n = cp.asarray(((n_coords - 1) / self.px_size), dtype=cp.float64)

        # Antenna coordinates (Fourier Domain - uvw coordinates)
        target_u = cp.asarray(
            2 * pi * (u_coords.flatten() * self.px_size),
            dtype=cp.float64,
        )
        target_v = cp.asarray(
            2 * pi * (v_coords.flatten() * self.px_size),
            dtype=cp.float64,
        )
        target_w = cp.asarray(
            2 * pi * (w_coords.flatten() * self.px_size),
            dtype=cp.float64,
        )

        if target_u:
            warnings.warn(
                "Some of the uvw coordinates lie outside the constructed image."
                "This can lead to cufinufft errors."
            )

        # Values at source position (Source intensities)
        c_values = cp.asarray(sky_values.flatten(), dtype=cp.complex128)

        result = self.ft(
            source_l,
            source_m,
            source_n,
            c_values,
            target_u,
            target_v,
            target_w,
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
        """Calculate the inverse fast Fourier transform with non-uniform source
        and non-uniform target coordinates.
        """
        # Antenna coordinates (Fourier Domain - uvw coordinates)
        source_u = cp.asarray(
            2 * pi * (u_coords.flatten() * self.px_size), dtype=cp.float64
        )
        source_v = cp.asarray(
            2 * pi * (v_coords.flatten() * self.px_size), dtype=cp.float64
        )
        source_w = cp.asarray(
            2 * pi * (w_coords.flatten() * self.px_size), dtype=cp.float64
        )

        # Fourier coeficients at antenna positions (Visibilities)
        c_values = cp.asarray(visibilities.flatten(), dtype=cp.complex128)

        # Sky coordinates (Image domain - lmn coordinates)
        target_l = cp.asarray((l_coords / self.px_size), dtype=cp.float64)
        target_m = cp.asarray((m_coords / self.px_size), dtype=cp.float64)
        target_n = cp.asarray(((n_coords - 1) / self.px_size), dtype=cp.float64)

        result = (
            self.ift(
                source_u, source_v, source_w, c_values, target_l, target_m, target_n
            )
            / self.px_scaling
        )

        if return_torch:
            sky_intensities = torch.as_tensor(result, device="cuda")
        else:
            sky_intensities = result.get()

        return sky_intensities
