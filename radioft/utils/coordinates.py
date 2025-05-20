import numpy as np
import torch


def create_lm_grid(fov_arcsec, img_size, source_declination):
    fov_rad = np.deg2rad(fov_arcsec / 3600, dtype=np.float64)

    # define resolution
    res = fov_rad / img_size

    dec = torch.deg2rad(torch.tensor(source_declination))

    r = torch.from_numpy(
        np.arange(
            start=-(img_size / 2) * res,
            stop=(img_size / 2) * res,
            step=res,
            dtype=np.float128,
        ).astype(np.float64)
    )
    d = r + dec

    R, _ = torch.meshgrid((r, r), indexing="ij")
    _, D = torch.meshgrid((d, d), indexing="ij")
    rd_grid = torch.cat([R[..., None], D[..., None]], dim=2)

    rd = rd_grid.numpy().astype(np.float128)

    dec = np.deg2rad(torch.tensor(source_declination).numpy()).astype(np.float64)

    lm_grid = np.zeros(rd.shape, dtype=np.float64)
    lm_grid[..., 0] = np.cos(rd[..., 1]) * np.sin(rd[..., 0])
    lm_grid[..., 1] = np.sin(rd[..., 1]) * np.cos(dec) - np.cos(rd[..., 1]) * np.sin(
        dec
    ) * np.cos(rd[..., 0])

    return torch.from_numpy(lm_grid.astype("float64"))
