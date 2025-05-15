import torch
from torch.autograd import Function
from radioft.cuda.kernels import compute_phase_matrix, compute_inverse_phase_matrix


class CudaDFTFunction(Function):
    @staticmethod
    def forward(
        ctx, sky_values, l_coords, m_coords, n_coords, u_coords, v_coords, w_coords
    ):
        # Save for backward
        ctx.save_for_backward(
            l_coords, m_coords, n_coords, u_coords, v_coords, w_coords
        )

        # Get phase matrix from CUDA kernel
        phase_matrix = compute_phase_matrix(
            l_coords, m_coords, n_coords, u_coords, v_coords, w_coords
        )

        # Compute trig functions
        cos_phase = torch.cos(phase_matrix)
        sin_phase = torch.sin(phase_matrix)

        # Extract real and imaginary parts of sky values
        sky_real = torch.real(sky_values)
        sky_imag = torch.imag(sky_values)

        # Handle batch dimension
        batch_size = sky_values.shape[0] if sky_values.dim() > 1 else 1
        if sky_values.dim() == 1:
            sky_real = sky_real.unsqueeze(0)
            sky_imag = sky_imag.unsqueeze(0)

        # Pre-allocate output tensors
        num_vis = phase_matrix.shape[0]
        vis_real = torch.zeros(
            (batch_size, num_vis), dtype=torch.float64, device=sky_values.device
        )
        vis_imag = torch.zeros(
            (batch_size, num_vis), dtype=torch.float64, device=sky_values.device
        )

        # Process each batch separately
        for b in range(batch_size):
            # Matrix multiply for each batch item
            vis_real[b] = (cos_phase @ sky_real[b]) - (sin_phase @ sky_imag[b])
            vis_imag[b] = (sin_phase @ sky_real[b]) + (cos_phase @ sky_imag[b])

        # Combine to complex
        visibilities = torch.complex(vis_real, vis_imag)

        # Remove batch dimension if input wasn't batched
        if sky_values.dim() == 1:
            visibilities = visibilities.squeeze(0)

        return visibilities

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        l_coords, m_coords, n_coords, u_coords, v_coords, w_coords = ctx.saved_tensors

        # Initialize gradients as None
        grad_sky = grad_l = grad_m = grad_n = grad_u = grad_v = grad_w = None

        # Handle batch dimension
        batch_size = grad_output.shape[0] if grad_output.dim() > 1 else 1
        unbatched_grad = grad_output.dim() == 1

        # We only care about gradient with respect to sky_values for the neural network
        if ctx.needs_input_grad[0]:
            # Get phase matrix again
            phase_matrix = compute_phase_matrix(
                l_coords, m_coords, n_coords, u_coords, v_coords, w_coords
            )

            # Compute trig functions
            cos_phase = torch.cos(phase_matrix)
            sin_phase = torch.sin(phase_matrix)

            # Extract real and imaginary parts of gradients
            grad_real = torch.real(grad_output)
            grad_imag = torch.imag(grad_output)

            if unbatched_grad:
                grad_real = grad_real.unsqueeze(0)
                grad_imag = grad_imag.unsqueeze(0)

            # Pre-allocate output tensors
            num_pixels = phase_matrix.shape[1]
            grad_sky_real = torch.zeros(
                (batch_size, num_pixels),
                dtype=l_coords.dtype,
                device=grad_output.device,
            )
            grad_sky_imag = torch.zeros(
                (batch_size, num_pixels),
                dtype=l_coords.dtype,
                device=grad_output.device,
            )

            # Process each batch separately
            for b in range(batch_size):
                # Compute gradients via transpose of forward pass operations
                grad_sky_real[b] = (cos_phase.T @ grad_real[b]) + (
                    sin_phase.T @ grad_imag[b]
                )
                grad_sky_imag[b] = (-sin_phase.T @ grad_real[b]) + (
                    cos_phase.T @ grad_imag[b]
                )

            # Combine to complex
            grad_sky = torch.complex(grad_sky_real, grad_sky_imag)

            # Remove batch dimension if input wasn't batched
            if unbatched_grad:
                grad_sky = grad_sky.squeeze(0)

        # Return gradients for all inputs
        return grad_sky, grad_l, grad_m, grad_n, grad_u, grad_v, grad_w


class CudaIDFTFunction(Function):
    @staticmethod
    def forward(
        ctx, visibilities, l_coords, m_coords, n_coords, u_coords, v_coords, w_coords
    ):
        """
        Simplified forward pass that computes inverse DFT for a single chunk
        without handling the overall chunking strategy
        """
        # Save for backward
        ctx.save_for_backward(
            l_coords, m_coords, n_coords, u_coords, v_coords, w_coords
        )

        # Handle batch dimension
        batch_size = visibilities.shape[0] if visibilities.dim() > 1 else 1
        if visibilities.dim() == 1:
            visibilities = visibilities.unsqueeze(0)

        # Extract dimensions
        num_vis = u_coords.shape[0]  # Number of visibility points in this chunk
        num_pixels = l_coords.shape[0]  # Number of pixel points in this chunk

        # Store for backward pass
        ctx.num_vis = num_vis
        ctx.num_pixels = num_pixels

        # Extract real and imaginary parts of visibilities
        vis_real = torch.real(visibilities)
        vis_imag = torch.imag(visibilities)

        # Pre-allocate output tensor for this chunk
        sky_values = torch.zeros(
            (batch_size, num_pixels),
            dtype=visibilities.dtype,
            device=visibilities.device,
        )

        # Compute phase matrix for this chunk
        phase_matrix = compute_inverse_phase_matrix(
            l_coords, m_coords, n_coords, u_coords, v_coords, w_coords
        )

        # Compute trig functions
        cos_phase = torch.cos(phase_matrix)
        sin_phase = torch.sin(phase_matrix)

        # Process each batch separately
        for b in range(batch_size):
            # Extract visibility data for this batch
            batch_vis_real = vis_real[b]  # [num_vis]
            batch_vis_imag = vis_imag[b]  # [num_vis]

            # Reshape for efficient matrix multiplication
            batch_vis_real_col = batch_vis_real.reshape(-1, 1)  # [num_vis, 1]
            batch_vis_imag_col = batch_vis_imag.reshape(-1, 1)  # [num_vis, 1]

            # Using efficient matrix operations
            # [num_pixels, num_vis] @ [num_vis, 1]
            real_contrib = torch.matmul(cos_phase.T, batch_vis_real_col) - torch.matmul(
                sin_phase.T, batch_vis_imag_col
            )
            imag_contrib = torch.matmul(sin_phase.T, batch_vis_real_col) + torch.matmul(
                cos_phase.T, batch_vis_imag_col
            )

            # Store result for this batch
            sky_values[b] = torch.complex(
                real_contrib.squeeze(), imag_contrib.squeeze()
            )

        # Remove batch dimension if input wasn't batched
        if visibilities.dim() == 1:
            sky_values = sky_values.squeeze(0)

        return sky_values

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for the simplified IDFT function
        """
        # Retrieve saved tensors
        l_coords, m_coords, n_coords, u_coords, v_coords, w_coords = ctx.saved_tensors
        num_vis = ctx.num_vis

        # Initialize gradients as None
        grad_vis = grad_l = grad_m = grad_n = grad_u = grad_v = grad_w = None

        # We only compute gradient for visibilities if required
        if ctx.needs_input_grad[0]:
            # Handle batch dimension
            batch_size = grad_output.shape[0] if grad_output.dim() > 1 else 1
            unbatched = grad_output.dim() == 1
            if unbatched:
                grad_output = grad_output.unsqueeze(0)

            # Pre-allocate gradient tensor for visibilities
            grad_vis = torch.zeros(
                (batch_size, num_vis),
                dtype=torch.complex128
                if grad_output.dtype == torch.complex128
                else torch.complex64,
                device=grad_output.device,
            )

            # Extract real and imaginary parts of grad_output
            grad_real = torch.real(grad_output)  # [batch_size, num_pixels]
            grad_imag = torch.imag(grad_output)  # [batch_size, num_pixels]

            # Compute phase matrix for gradient computation (use forward phase)
            # For backward pass through IDFT, we need the complex conjugate
            # of the forward phase factors (which is the regular DFT phase)
            phase_matrix = compute_phase_matrix(
                l_coords, m_coords, n_coords, u_coords, v_coords, w_coords
            )

            # Compute trig functions
            cos_phase = torch.cos(phase_matrix)  # [num_vis, num_pixels]
            sin_phase = torch.sin(phase_matrix)  # [num_vis, num_pixels]

            # Process each batch
            for b in range(batch_size):
                # Get gradients for this batch
                batch_grad_real = grad_real[b]  # [num_pixels]
                batch_grad_imag = grad_imag[b]  # [num_pixels]

                # Using matrix multiplication to compute gradients
                # [num_vis, num_pixels] @ [num_pixels]
                real_grad = torch.matmul(cos_phase, batch_grad_real) + torch.matmul(
                    sin_phase, batch_grad_imag
                )
                imag_grad = torch.matmul(-sin_phase, batch_grad_real) + torch.matmul(
                    cos_phase, batch_grad_imag
                )

                # Store in gradient tensor
                grad_vis[b] = torch.complex(real_grad, imag_grad)

            # Remove batch dimension if needed
            if unbatched:
                grad_vis = grad_vis.squeeze(0)

        # Return gradients for all inputs
        return grad_vis, grad_l, grad_m, grad_n, grad_u, grad_v, grad_w


# Create wrapper functions for ease of use
def cuda_dft(sky_values, l_coords, m_coords, n_coords, u_coords, v_coords, w_coords):
    """
    CUDA-accelerated DFT without chunking - for use by HybridPyTorchCudaDFT
    """
    return CudaDFTFunction.apply(
        sky_values, l_coords, m_coords, n_coords, u_coords, v_coords, w_coords
    )


def cuda_idft(visibilities, l_coords, m_coords, n_coords, u_coords, v_coords, w_coords):
    """
    CUDA-accelerated IDFT without chunking - for use by HybridPyTorchCudaDFT
    """
    return CudaIDFTFunction.apply(
        visibilities, l_coords, m_coords, n_coords, u_coords, v_coords, w_coords
    )
