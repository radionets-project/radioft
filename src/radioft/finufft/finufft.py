import warnings
from functools import partial
from math import pi
from operator import itemgetter

import cufinufft
import cupy as cp
import torch

uvw_map = {
    0: "u",
    1: "v",
    2: "w",
}


def cupy_to_torch(x_cp):
    """CuPy array -> Torch tensor on GPU (Zero-Copy via DLPack)"""
    return torch.utils.dlpack.from_dlpack(x_cp.toDlpack())


def torch_to_cupy(x_torch):
    """Torch tensor -> CuPy array on GPU (Zero-Copy via DLPack)"""
    return cp.fromDlpack(torch.utils.dlpack.to_dlpack(x_torch))


class CupyFinufft:
    """Wraper to use Finufft Type 3d3 for radio interferometry data."""

    def __init__(
        self,
        image_size,
        fov_arcsec,
        eps=1e-12,
    ):
        """Wraper to use Finufft Type 3d3 for radio interferometry data."""
        self.px_size = ((fov_arcsec / 3600) * pi / 180) / image_size
        self.px_scaling = image_size**2
        self.image_size = image_size

        self.ft = partial(cufinufft.nufft3d3, isign=-1, eps=eps)
        self.ift = partial(cufinufft.nufft3d3, isign=+1, eps=eps)

    def _compute_visibility_weights_and_indices(
        self,
        u_coords,
        v_coords,
        w_coords,
        image_size=None,
    ):
        """
        Compute visibility weights and bin indices for each visibility sample.

        This method maps each visibility to the corresponding image pixel and
        returns both the bin count for that pixel and the indices needed for
        normalization.

        Parameters
        ----------
        u_coords : torch.Tensor
            U coordinates of visibility samples
        v_coords : torch.Tensor
            V coordinates of visibility samples
        w_coords : torch.Tensor
            W coordinates of visibility samples (not used for 2D binning)
        image_size : int, optional
            Size of the output image. If None, uses self.image_size

        Returns
        -------
        visibility_weights : torch.Tensor
            1D tensor of length num_visibilities containing the bin count
            for each visibility sample (i.e., how many visibilities map to
            the same pixel as this visibility)
        """
        if image_size is None:
            image_size = self.image_size

        # Convert UV coordinates to pixel indices
        u_pixels = cp.round(u_coords / (2 * pi / image_size)).astype(cp.float64)
        v_pixels = cp.round(v_coords / (2 * pi / image_size)).astype(cp.float64)

        # Create validity mask BEFORE centering
        u_min = -image_size // 2
        u_max = image_size // 2
        valid_mask = (
            (u_pixels >= u_min)
            & (u_pixels < u_max)
            & (v_pixels >= u_min)
            & (v_pixels < u_max)
        )

        # Shift to center the image FIRST, then clip
        u_pixels = u_pixels + image_size // 2
        v_pixels = v_pixels + image_size // 2

        # Clip to valid image bounds (after centering)
        u_pixels = cp.clip(u_pixels, 0, image_size - 1)
        v_pixels = cp.clip(v_pixels, 0, image_size - 1)

        # Convert to integer indices
        u_pixels = u_pixels.astype(cp.int64)
        v_pixels = v_pixels.astype(cp.int64)

        # Convert 2D indices to 1D
        linear_indices = v_pixels * image_size + u_pixels

        # Compute histogram
        histogram_flat = cp.bincount(
            linear_indices, minlength=image_size * image_size
        ).astype(cp.float64)

        # Map each visibility to its bin count
        visibility_weights = histogram_flat[linear_indices]

        # Handle invalid entries - set to 1.0, NOT 0
        visibility_weights[~valid_mask] = 1.0

        return visibility_weights

    def nufft(
        self,
        sky_values,
        l_coords,
        m_coords,
        n_coords,
        u_coords,
        v_coords,
        w_coords,
    ):
        """Calculate the fast Fourier transform with non-uniform source
        and non-uniform target coordinates.
        """
        # Sky coordinates (Image domain - lmn coordinates)
        source_l = torch_to_cupy(l_coords / self.px_size).astype(cp.float64)
        source_m = torch_to_cupy(m_coords / self.px_size).astype(cp.float64)
        source_n = torch_to_cupy((n_coords - 1) / self.px_size).astype(cp.float64)

        # Antenna coordinates (Fourier Domain - uvw coordinates)
        target_u = torch_to_cupy(2 * pi * (u_coords.flatten() * self.px_size)).astype(
            cp.float64
        )
        target_v = torch_to_cupy(2 * pi * (v_coords.flatten() * self.px_size)).astype(
            cp.float64
        )
        target_w = torch_to_cupy(2 * pi * (w_coords.flatten() * self.px_size)).astype(
            cp.float64
        )

        outside_bounds = cp.array(
            [
                (target_u <= -cp.pi) | (target_u > cp.pi),
                (target_v <= -cp.pi) | (target_v > cp.pi),
                (target_w <= -cp.pi) | (target_w > cp.pi),
            ]
        )
        coord_outside = cp.where(cp.any(outside_bounds, axis=1))[0]
        if outside_bounds.any():
            warnings.warning(
                f"Some of the {', '.join(itemgetter(*coord_outside.get())(uvw_map))} "
                "coordinates lie outside the constructed image. This can lead to "
                "cufinufft errors."
            )

        # Values at source position (Source intensities)
        c_values = torch_to_cupy(sky_values.flatten()).astype(cp.complex128)

        result = self.ft(
            source_l,
            source_m,
            source_n,
            c_values,
            target_u,
            target_v,
            target_w,
        )

        visibilities = cupy_to_torch(result)

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
    ):
        """Calculate the inverse fast Fourier transform with non-uniform source
        and non-uniform target coordinates.
        """
        # Antenna coordinates (Fourier Domain - uvw coordinates)
        source_u = torch_to_cupy(2 * pi * (u_coords.flatten() * self.px_size)).astype(
            cp.float64
        )
        source_v = torch_to_cupy(2 * pi * (v_coords.flatten() * self.px_size)).astype(
            cp.float64
        )
        source_w = torch_to_cupy(2 * pi * (w_coords.flatten() * self.px_size)).astype(
            cp.float64
        )

        # Compute visibility weights: for each visibility, how many other
        # visibilities fall into the same UV bin?
        visibility_weights = self._compute_visibility_weights_and_indices(
            source_u,
            source_v,
            source_w,
            image_size=self.image_size,
        )

        outside_bounds = cp.array(
            [
                (source_u <= -cp.pi) | (source_u > cp.pi),
                (source_v <= -cp.pi) | (source_v > cp.pi),
                (source_w <= -cp.pi) | (source_w > cp.pi),
            ]
        )
        coord_outside = cp.where(cp.any(outside_bounds, axis=1))[0]

        if outside_bounds.any():
            warnings.warning(
                f"Some of the {', '.join(itemgetter(*coord_outside.get())(uvw_map))} "
                "coordinates lie outside the constructed image. This can lead to "
                "cufinufft errors."
            )

        # Fourier coeficients at antenna positions (Visibilities)
        c_values = torch_to_cupy(visibilities.flatten()).astype(cp.complex128)

        # Normalize visibility values by dividing by their bin counts
        # This means visibilities that fall into the same bin are averaged
        c_values_normalized = c_values / visibility_weights

        # Sky coordinates (Image domain - lmn coordinates)
        target_l = torch_to_cupy(l_coords.flatten() / self.px_size).astype(cp.float64)
        target_m = torch_to_cupy(m_coords.flatten() / self.px_size).astype(cp.float64)
        target_n = torch_to_cupy((n_coords.flatten() - 1) / self.px_size).astype(
            cp.float64
        )

        result = (
            self.ift(
                source_u,
                source_v,
                source_w,
                c_values_normalized,
                target_l,
                target_m,
                target_n,
            )
            / self.px_scaling
        )

        sky_intensities = cupy_to_torch(result)

        return sky_intensities
