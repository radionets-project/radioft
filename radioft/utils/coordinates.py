import numpy as np
import torch
from astropy.constants import R_earth, c


def create_lm_grid(fov_arcsec, img_size, source_declination):
    fov_rad = np.deg2rad(fov_arcsec / 3600, dtype=np.float128)

    # define resolution
    res = fov_rad / img_size

    dec = np.deg2rad(source_declination)

    r = np.arange(
        start=-(img_size / 2) * res,
        stop=(img_size / 2) * res,
        step=res,
        dtype=np.float128,
    )
    d = r + dec

    R, _ = np.meshgrid(r, r, indexing="ij")
    _, D = np.meshgrid(d, d, indexing="ij")
    rd = np.concatenate([R[..., None], D[..., None]], axis=2)

    lm_grid = np.zeros(rd.shape, dtype=np.float128)
    lm_grid[..., 0] = np.cos(rd[..., 1]) * np.sin(rd[..., 0])
    lm_grid[..., 1] = np.sin(rd[..., 1]) * np.cos(dec) - np.cos(rd[..., 1]) * np.sin(
        dec
    ) * np.cos(rd[..., 0])

    return torch.from_numpy(lm_grid.astype(np.float64))


def create_uvw_dense(N, fov, freq, earth_radius=R_earth.value):
    fov_arc = np.deg2rad(fov / 3600, dtype=np.float128)
    delta = fov_arc ** (-1)

    u_dense = np.arange(
        start=-(N / 2) * delta,
        stop=(N / 2) * delta,
        step=delta,
        dtype=np.float128,
    )

    v_dense = u_dense

    uu, vv = np.meshgrid(u_dense, v_dense, indexing="ij")

    # Calculate distance from center in uv-plane (in wavelengths)
    r_squared = uu**2 + vv**2

    wavelength = c.value / freq

    # Convert uv distances to physical distances
    physical_distance = r_squared * wavelength

    # Calculate w coordinate based on Earth curvature
    # Using the approximation w ≈ d²/2R, where d is the distance in the uv-plane
    # and R is the Earth radius in wavelengths
    earth_radius_wavelengths = earth_radius / wavelength
    ww = physical_distance / (2 * earth_radius_wavelengths)

    return (
        torch.from_numpy(uu.astype(np.float64)),
        torch.from_numpy(vv.astype(np.float64)),
        torch.from_numpy(ww.astype(np.float64)),
    )
