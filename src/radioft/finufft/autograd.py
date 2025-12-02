import torch


class FINUFFTAutograd(torch.autograd.Function):
    """
    Custom autograd function that uses finite differences for NUFFT gradient.
    """

    @staticmethod
    def forward(ctx, image, lmn_coords, uvw_coords, finufft):
        """
        Forward pass: compute NUFFT normally.
        """
        # Save for backward
        ctx.save_for_backward(image, lmn_coords, uvw_coords)
        ctx.finufft = finufft

        # Compute NUFFT (no gradient tracking)
        with torch.no_grad():
            # Use complex128 (double precision) instead of complex64
            vis = finufft.nufft(
                image.detach().to(torch.complex128),
                lmn_coords[0],
                lmn_coords[1],
                lmn_coords[2],
                uvw_coords[0],
                uvw_coords[1],
                uvw_coords[2],
                return_torch=True,
            )

        return vis

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: use adjoint (inverse) NUFFT as approximate gradient.
        """
        image, lmn_coords, uvw_coords = ctx.saved_tensors
        finufft = ctx.finufft

        # Use inverse NUFFT as approximate adjoint
        with torch.no_grad():
            grad_image = finufft.inufft(
                grad_output.detach().to(torch.complex128),
                lmn_coords[0],
                lmn_coords[1],
                lmn_coords[2],
                uvw_coords[0],
                uvw_coords[1],
                uvw_coords[2],
                return_torch=True,
            )

            # Take real part since input was real
            grad_image = torch.real(grad_image).reshape(image.shape)

            # Convert to same dtype as input image (double)
            grad_image = grad_image.to(image.dtype)

        return grad_image, None, None, None


def apply_nufft_differentiable(image, lmn_coords, uvw_coords, finufft):
    """
    Apply NUFFT with gradient support.
    """
    return FINUFFTAutograd.apply(image, lmn_coords, uvw_coords, finufft)


class IFINUFFTAutograd(torch.autograd.Function):
    """
    Custom autograd function for differentiable inverse NUFFT (double precision).
    """

    @staticmethod
    def forward(ctx, vis, lmn_coords, uvw_coords, finufft, image_shape):
        """
        Forward pass: compute inverse NUFFT normally.

        Args:
            vis: Complex visibilities (1D, complex128)
            lmn_coords: Sky coordinates (3, N) - double
            uvw_coords: UV coordinates (3, M) - double
            finufft: FINUFFT object
            image_shape: Output image shape (H, W)
        """
        # Save for backward
        ctx.save_for_backward(vis, lmn_coords, uvw_coords)
        ctx.finufft = finufft
        ctx.image_shape = image_shape

        # Compute inverse NUFFT (no gradient tracking)
        with torch.no_grad():
            # Use complex128 (double precision)
            image = finufft.inufft(
                vis.detach().to(torch.complex128),
                lmn_coords[0],
                lmn_coords[1],
                lmn_coords[2],
                uvw_coords[0],
                uvw_coords[1],
                uvw_coords[2],
            )

            # Reshape to image
            if len(image_shape) == 2:
                image = image.reshape(image_shape)

        return image

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: use forward NUFFT as adjoint.

        The adjoint of inverse NUFFT is forward NUFFT.
        """
        vis, lmn_coords, uvw_coords = ctx.saved_tensors
        finufft = ctx.finufft

        # Use forward NUFFT as adjoint
        with torch.no_grad():
            # Flatten gradient if needed
            grad_flat = grad_output.flatten()

            # Apply forward NUFFT to gradient
            grad_vis = finufft.nufft(
                grad_flat.detach().to(torch.complex128),
                lmn_coords[0],
                lmn_coords[1],
                lmn_coords[2],
                uvw_coords[0],
                uvw_coords[1],
                uvw_coords[2],
            )

            # Convert to same dtype as input vis (complex128)
            grad_vis = grad_vis.to(vis.dtype)

        # Return gradients: (vis, lmn_coords, uvw_coords, finufft, image_shape)
        return grad_vis, None, None, None, None


def apply_inufft_differentiable(vis, lmn_coords, uvw_coords, finufft, image_shape):
    """
    Apply inverse NUFFT with gradient support (double precision).

    Args:
        vis: Complex visibilities (complex128)
        lmn_coords: Sky coordinates (3, N) - double
        uvw_coords: UV coordinates (3, M) - double
        finufft: FINUFFT object
        image_shape: Output shape (H, W) or (H*W,)

    Returns:
        image: Complex image (complex128)
    """
    return IFINUFFTAutograd.apply(vis, lmn_coords, uvw_coords, finufft, image_shape)
