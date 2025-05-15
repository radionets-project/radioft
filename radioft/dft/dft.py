import torch
from .utils import cuda_dft, cuda_idft
from radioft.utils.sizes import get_optimal_chunk_sizes


class HybridPyTorchCudaDFT:
    """
    Deterministic hybrid PyTorch-CUDA DFT implementation with performance optimizations
    """

    def __init__(self, device="cuda", benchmark=False):
        self.device = device

        # Set deterministic algorithms for more consistent performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = benchmark

        self.dft = cuda_dft
        self.idft = cuda_idft

    def forward(
        self,
        sky_values,
        l_coords,
        m_coords,
        n_coords,
        u_coords,
        v_coords,
        w_coords,
        max_memory_gb=4,
    ):
        """
        Compute DFT with predictable performance
        """
        # Start with empty cache for more deterministic behavior
        torch.cuda.empty_cache()

        # Convert to appropriate format
        sky_values = sky_values.to(self.device).cdouble()
        l_coords = l_coords.to(self.device).double()
        m_coords = m_coords.to(self.device).double()
        n_coords = n_coords.to(self.device).double()
        u_coords = u_coords.to(self.device).double()
        v_coords = v_coords.to(self.device).double()
        w_coords = w_coords.to(self.device).double()

        # Handle batched or unbatched input
        if sky_values.dim() == 1:
            unbatched_input = True
            sky_values = sky_values.unsqueeze(0)
        else:
            unbatched_input = False

        batch_size = sky_values.shape[0]
        num_pixels = len(l_coords)
        num_vis = len(u_coords)

        # Get optimized chunk sizes
        vis_chunk_size, pixel_chunk_size = get_optimal_chunk_sizes(
            num_pixels, num_vis, max_memory_gb
        )

        # Pre-allocate output with zeros
        visibilities = torch.zeros(
            (batch_size, num_vis), dtype=torch.complex128, device=self.device
        )

        # Calculate total chunks for progress bar
        total_chunks = ((num_vis + vis_chunk_size - 1) // vis_chunk_size) * (
            (num_pixels + pixel_chunk_size - 1) // pixel_chunk_size
        )

        # Use fixed seed for more deterministic behavior
        torch.manual_seed(0)

        # Process visibility points in chunks
        for vis_start in range(0, num_vis, vis_chunk_size):
            vis_end = min(vis_start + vis_chunk_size, num_vis)
            vis_chunk_len = vis_end - vis_start

            # Get current visibility chunk
            u_chunk = u_coords[
                vis_start:vis_end
            ].contiguous()  # Force contiguous for better performance
            v_chunk = v_coords[vis_start:vis_end].contiguous()
            w_chunk = w_coords[vis_start:vis_end].contiguous()

            # Pre-allocate accumulators for this visibility chunk - more efficient
            chunk_vis = torch.zeros(
                (batch_size, vis_chunk_len), dtype=torch.complex128, device=self.device
            )

            # Process pixels in chunks
            for pixel_start in range(0, num_pixels, pixel_chunk_size):
                pixel_end = min(pixel_start + pixel_chunk_size, num_pixels)
                pixel_chunk_len = pixel_end - pixel_start

                # Get current pixel chunk - force contiguous for better performance
                l_chunk = l_coords[pixel_start:pixel_end].contiguous()
                m_chunk = m_coords[pixel_start:pixel_end].contiguous()
                n_chunk = n_coords[pixel_start:pixel_end].contiguous()

                sky_values_chunk = sky_values[:, pixel_start:pixel_end]
                chunk_vis += self.dft(
                    sky_values_chunk,
                    l_chunk,
                    m_chunk,
                    n_chunk,
                    u_chunk,
                    v_chunk,
                    w_chunk,
                )
            # Store result for this visibility chunk
            visibilities[:, vis_start:vis_end] = chunk_vis

            # Clean up
            del chunk_vis

        # Remove batch dimension if input was not batched
        if unbatched_input:
            visibilities = visibilities.squeeze(0)

        return visibilities

    def inverse(
        self,
        visibilities,
        l_coords,
        m_coords,
        n_coords,
        u_coords,
        v_coords,
        w_coords,
        max_memory_gb=20,
    ):
        """
        Compute inverse DFT with built-in chunking like the forward method
        """
        # Clear cache before starting
        torch.cuda.empty_cache()

        # Convert to appropriate format
        visibilities = visibilities.to(self.device).cdouble()
        l_coords = l_coords.to(self.device).double()
        m_coords = m_coords.to(self.device).double()
        n_coords = n_coords.to(self.device).double()
        u_coords = u_coords.to(self.device).double()
        v_coords = v_coords.to(self.device).double()
        w_coords = w_coords.to(self.device).double()

        # Handle batched or unbatched input
        if visibilities.dim() == 1:
            unbatched_input = True
            visibilities = visibilities.unsqueeze(0)
        else:
            unbatched_input = False

        # Extract dimensions
        batch_size = visibilities.shape[0]
        num_pixels = l_coords.shape[0]
        num_vis = u_coords.shape[0]

        # Get optimized chunk sizes
        vis_chunk_size, pixel_chunk_size = get_optimal_chunk_sizes(
            num_pixels, num_vis, max_memory_gb
        )

        # Pre-allocate output tensor for sky image
        sky_values = torch.zeros(
            (batch_size, num_pixels), dtype=torch.complex128, device=self.device
        )

        # Extract real and imaginary parts of visibilities
        vis_real = torch.real(visibilities)
        vis_imag = torch.imag(visibilities)

        # Process pixel points in chunks
        for pixel_start in range(0, num_pixels, pixel_chunk_size):
            pixel_end = min(pixel_start + pixel_chunk_size, num_pixels)
            pixel_chunk_len = pixel_end - pixel_start

            # Get current pixel chunk coordinates
            l_chunk = l_coords[pixel_start:pixel_end].contiguous()
            m_chunk = m_coords[pixel_start:pixel_end].contiguous()
            n_chunk = n_coords[pixel_start:pixel_end].contiguous()

            # Pre-allocate accumulators for this visibility chunk - more efficient
            sky_values_chunk = torch.zeros(
                (batch_size, pixel_chunk_len),
                dtype=torch.complex128,
                device=self.device,
            )

            # Process visibility points in sub-chunks
            for vis_start in range(0, num_vis, vis_chunk_size):
                vis_end = min(vis_start + vis_chunk_size, num_vis)
                vis_chunk_len = vis_end - vis_start

                # Get current visibility chunk
                u_chunk = u_coords[vis_start:vis_end].contiguous()
                v_chunk = v_coords[vis_start:vis_end].contiguous()
                w_chunk = w_coords[vis_start:vis_end].contiguous()

                # Get visibility data for this chunk
                vis_chunk = visibilities[:, vis_start:vis_end]

                sky_values_chunk += self.idft(
                    vis_chunk, l_chunk, m_chunk, n_chunk, u_chunk, v_chunk, w_chunk
                )

            # Add contribution to the sky values
            sky_values[:, pixel_start:pixel_end] += sky_values_chunk

            # Free memory
            torch.cuda.empty_cache()
            del sky_values_chunk

        # Normalize by number of visibility points
        sky_values = sky_values / num_vis

        # Remove batch dimension if input wasn't batched
        if unbatched_input:
            sky_values = sky_values.squeeze(0)

        return sky_values


class OptimizedPyTorchDFT:
    """
    Optimized PyTorch DFT implementation for radio astronomy visibilities

    This implementation achieves identical results to the standard PyTorch
    implementation but with improved performance through careful optimization
    of computation patterns and memory usage.

    Numerical precision is maintained at the level of PyTorch's native operations
    (typically ~10^-14 to 10^-15).

    Parameters
    ----------
    device : str, optional
        Device to perform calculations on ('cuda' or 'cpu'). Default: 'cuda'
    chunk_size : int, optional
        Maximum chunk size for processing to manage memory usage. Default: 5000
    """

    def __init__(self, device: str = "cuda", chunk_size: int = 5000) -> None:
        """
        Initialize the DFT calculator

        Parameters
        ----------
        device: str, optional
            Device to perform calculations on ('cuda' or 'cpu'). Default: 'cuda'
        chunk_size: int, optional
            Maximum chunk size for processing to manage memory usage. Default: 5000
        """
        self.device = device
        self.chunk_size = chunk_size

    def __call__(
        self, sky_values, l_coords, m_coords, n_coords, u_coords, v_coords, w_coords
    ):
        """
        Compute DFT with optimized PyTorch operations

        Parameters
        ----------
        sky_values :
            Complex sky brightness values [num_pixels] or [batch, num_pixels]
        l_coords :
            Sky l-direction coordinates [num_pixels]
        m_coords :
            Sky m-direction coordinates [num_pixels]
        n_coords :
            Sky n-direction coordinates [num_pixels]
        u_coords :
            UV u-coordinates [num_vis]
        v_coords :
            UV v-coordinates [num_vis]
        w_coords :
            UV w-coordinates [num_vis]

        Returns
        -------
        visibilities :
            Computed visibility values [num_vis] or [batch, num_vis]
        """
        # Convert to appropriate format
        sky_values = sky_values.to(self.device).cdouble()
        l_coords = l_coords.to(self.device).double()
        m_coords = m_coords.to(self.device).double()
        n_coords = n_coords.to(self.device).double()
        u_coords = u_coords.to(self.device).double()
        v_coords = v_coords.to(self.device).double()
        w_coords = w_coords.to(self.device).double()

        # Handle batched or unbatched input
        if sky_values.dim() == 1:
            unbatched_input = True
            sky_values = sky_values.unsqueeze(0)
        else:
            unbatched_input = False

        batch_size = sky_values.shape[0]
        num_pixels = len(l_coords)
        num_vis = len(u_coords)

        # Initialize output visibilities
        visibilities = torch.zeros(
            (batch_size, num_vis), dtype=torch.complex128, device=self.device
        )

        # Extract real and imaginary parts once (avoids repeated extraction)
        sky_real_parts = torch.real(sky_values)
        sky_imag_parts = torch.imag(sky_values)

        # Get adjusted n_coords (n-1) used in calculations
        n_minus_one = n_coords - 1.0

        # Process in optimized chunks to maximize throughput
        vis_chunk_size = min(self.chunk_size, num_vis)
        pixel_chunk_size = min(self.chunk_size, num_pixels)

        # Process visibility points in chunks
        for vis_start in range(0, num_vis, vis_chunk_size):
            vis_end = min(vis_start + vis_chunk_size, num_vis)

            # Get current visibility chunk
            u_chunk = u_coords[vis_start:vis_end]
            v_chunk = v_coords[vis_start:vis_end]
            w_chunk = w_coords[vis_start:vis_end]

            # Process pixel chunks for each visibility chunk
            for pixel_start in range(0, num_pixels, pixel_chunk_size):
                pixel_end = min(pixel_start + pixel_chunk_size, num_pixels)

                # Get current pixel chunk
                l_chunk = l_coords[pixel_start:pixel_end]
                m_chunk = m_coords[pixel_start:pixel_end]
                n_chunk_minus_one = n_minus_one[pixel_start:pixel_end]

                # Compute phase matrix efficiently using outer products
                # This is equivalent to matrix multiplication but more direct
                # Shape: [vis_chunk_size, pixel_chunk_size]
                u_term = torch.outer(u_chunk, l_chunk)
                v_term = torch.outer(v_chunk, m_chunk)
                w_term = torch.outer(w_chunk, n_chunk_minus_one)

                # Combine terms and compute phase
                # Using a single combined phase calculation avoids intermediate allocations
                phase = -2.0 * torch.pi * (u_term + v_term + w_term)

                # Pre-compute trig functions (more efficient than complex exponentiation)
                cos_phase = torch.cos(phase)
                sin_phase = torch.sin(phase)

                # Process each batch
                for b in range(batch_size):
                    # Get current sky values
                    sky_real = sky_real_parts[b, pixel_start:pixel_end]
                    sky_imag = sky_imag_parts[b, pixel_start:pixel_end]

                    # Matrix-vector multiplications for real and imaginary parts
                    # These operations are highly optimized in PyTorch
                    real_contribution = (cos_phase @ sky_real) - (sin_phase @ sky_imag)
                    imag_contribution = (sin_phase @ sky_real) + (cos_phase @ sky_imag)

                    # Combine into complex visibility
                    vis_contribution = torch.complex(
                        real_contribution, imag_contribution
                    )

                    # Add to result
                    visibilities[b, vis_start:vis_end] += vis_contribution

        # Remove batch dimension if input was not batched
        if unbatched_input:
            visibilities = visibilities.squeeze(0)

        return visibilities


class ChunkedDFT(torch.nn.Module):
    def __init__(self, chunk_size=5000, vis_chunk_size=10000, device="cuda"):
        super().__init__()
        self.chunk_size = chunk_size
        self.vis_chunk_size = vis_chunk_size
        self.device = device

    def forward(
        self, sky_values, l_coords, m_coords, n_coords, u_coords, v_coords, w_coords
    ):
        """
        Chunked DFT computation with non-uniform l,m coordinates

        Parameters
        ----------
        sky_values :
            Complex sky brightness values [batch, n_pixels] or [n_pixels]
        l_coords :
            Non-uniform image domain coordinates
        m_coords :
            Non-uniform image domain coordinates
        u_coords :
            Non-uniform sampling coordinates
        v_coords :
            Non-uniform sampling coordinates

        Returns
        -------
        visibilities :
            Complex visibilities [batch, n_vis] or [n_vis]
        """
        device = self.device

        # Check device of input variables
        sky_values = sky_values.to(device).cdouble()
        l_coords = l_coords.to(device)
        m_coords = m_coords.to(device)
        n_coords = n_coords.to(device)
        u_coords = u_coords.to(device)
        v_coords = v_coords.to(device)
        w_coords = w_coords.to(device)

        # Handle batched or unbatched input
        if sky_values.dim() == 1:
            # Single input, no batch
            batch_size = 1
            sky_values = sky_values.unsqueeze(0)  # Add batch dimension
            unbatched_input = True
        else:
            batch_size = sky_values.shape[0]
            unbatched_input = False

        num_pixels = len(l_coords)
        num_vis = len(u_coords)

        # Initialize output visibilities
        visibilities = torch.zeros(
            (batch_size, num_vis), dtype=torch.complex128, device=device
        )

        # Process visibility points in chunks
        for vis_start in range(0, num_vis, self.vis_chunk_size):
            vis_end = min(vis_start + self.vis_chunk_size, num_vis)
            u_chunk = u_coords[vis_start:vis_end]
            v_chunk = v_coords[vis_start:vis_end]
            w_chunk = w_coords[vis_start:vis_end]
            chunk_size = len(u_chunk)

            # Process pixel chunks for each visibility chunk
            for pixel_start in range(0, num_pixels, self.chunk_size):
                pixel_end = min(pixel_start + self.chunk_size, num_pixels)

                # Get current pixel chunk
                l_chunk = l_coords[pixel_start:pixel_end]
                m_chunk = m_coords[pixel_start:pixel_end]
                n_chunk = n_coords[pixel_start:pixel_end]

                # Use broadcasting to calculate phase for all combinations
                # [chunk_vis, 1] × [1, chunk_pixels]
                u_term = u_chunk.reshape(-1, 1) @ l_chunk.reshape(
                    1, -1
                )  # [chunk_vis, chunk_pixels]
                v_term = v_chunk.reshape(-1, 1) @ m_chunk.reshape(
                    1, -1
                )  # [chunk_vis, chunk_pixels]
                w_term = w_chunk.reshape(-1, 1) @ (
                    n_chunk.reshape(1, -1) - 1
                )  # [chunk_vis, chunk_pixels]

                # Calculate phase
                phase = (
                    -2 * np.pi * (u_term + v_term + w_term)
                )  # [chunk_vis, chunk_pixels]
                exponential = torch.exp(1j * phase)  # [chunk_vis, chunk_pixels]

                # Process each batch
                for b in range(batch_size):
                    # Extract sky values for this chunk
                    sky_chunk = sky_values[b][pixel_start:pixel_end]  # [chunk_pixels]

                    # Calculate contribution from this pixel chunk to all visibilities in current chunk
                    # [chunk_vis, chunk_pixels] × [chunk_pixels] → [chunk_vis]
                    vis_contribution = exponential @ sky_chunk

                    # Add to result
                    visibilities[b, vis_start:vis_end] += vis_contribution

        # Remove batch dimension if input was not batched
        if unbatched_input:
            visibilities = visibilities.squeeze(0)

        return visibilities


class ChunkedDFT_sincos(torch.nn.Module):
    def __init__(self, chunk_size=5000, vis_chunk_size=10000, device="cuda"):
        super().__init__()
        self.chunk_size = chunk_size
        self.vis_chunk_size = vis_chunk_size
        self.device = device

    def forward(
        self, sky_values, l_coords, m_coords, n_coords, u_coords, v_coords, w_coords
    ):
        """
        Chunked DFT computation with non-uniform l,m coordinates using cos and sin

        Parameters
        ----------
        sky_values :
            Complex sky brightness values [batch, n_pixels] or [n_pixels]
        l_coords :
            Non-uniform image domain coordinates
        m_coords :
            Non-uniform image domain coordinates
        u_coords :
            Non-uniform sampling coordinates
        v_coords :
            Non-uniform sampling coordinates

        Returns
        -------
        visibilities :
            Complex visibilities [batch, n_vis] or [n_vis]
        """
        device = self.device

        # Check device of input variables
        sky_values = sky_values.to(device).cdouble()
        l_coords = l_coords.to(device)
        m_coords = m_coords.to(device)
        n_coords = n_coords.to(device)
        u_coords = u_coords.to(device)
        v_coords = v_coords.to(device)
        w_coords = w_coords.to(device)

        # Handle batched or unbatched input
        if sky_values.dim() == 1:
            # Single input, no batch
            batch_size = 1
            sky_values = sky_values.unsqueeze(0)  # Add batch dimension
            unbatched_input = True
        else:
            batch_size = sky_values.shape[0]
            unbatched_input = False

        num_pixels = len(l_coords)
        num_vis = len(u_coords)

        # Initialize output visibilities
        visibilities = torch.zeros(
            (batch_size, num_vis), dtype=torch.complex128, device=device
        )

        # Process visibility points in chunks
        for vis_start in range(0, num_vis, self.vis_chunk_size):
            vis_end = min(vis_start + self.vis_chunk_size, num_vis)
            u_chunk = u_coords[vis_start:vis_end]
            v_chunk = v_coords[vis_start:vis_end]
            w_chunk = w_coords[vis_start:vis_end]
            chunk_size = len(u_chunk)

            # Process pixel chunks for each visibility chunk
            for pixel_start in range(0, num_pixels, self.chunk_size):
                pixel_end = min(pixel_start + self.chunk_size, num_pixels)

                # Get current pixel chunk
                l_chunk = l_coords[pixel_start:pixel_end]
                m_chunk = m_coords[pixel_start:pixel_end]
                n_chunk = n_coords[pixel_start:pixel_end]

                # Use broadcasting to calculate phase for all combinations
                # [chunk_vis, 1] × [1, chunk_pixels]
                u_term = u_chunk.reshape(-1, 1) @ l_chunk.reshape(
                    1, -1
                )  # [chunk_vis, chunk_pixels]
                v_term = v_chunk.reshape(-1, 1) @ m_chunk.reshape(
                    1, -1
                )  # [chunk_vis, chunk_pixels]
                w_term = w_chunk.reshape(-1, 1) @ (
                    n_chunk.reshape(1, -1) - 1
                )  # [chunk_vis, chunk_pixels]

                # Calculate phase
                phase = (
                    -2 * np.pi * (u_term + v_term + w_term)
                )  # [chunk_vis, chunk_pixels]

                # Calculate real and imaginary parts of the complex exponential
                cos_phase = torch.cos(phase)  # Real part
                sin_phase = torch.sin(phase)  # Imaginary part

                # Process each batch
                for b in range(batch_size):
                    # Extract sky values for this chunk
                    sky_chunk = sky_values[b][pixel_start:pixel_end]  # [chunk_pixels]

                    # Separate real and imaginary parts of sky values
                    sky_real = sky_chunk.real  # [chunk_pixels]
                    sky_imag = sky_chunk.imag  # [chunk_pixels]

                    # Calculate contributions using cos and sin
                    real_contribution = (cos_phase @ sky_real) - (
                        sin_phase @ sky_imag
                    )  # [chunk_vis]
                    imag_contribution = (sin_phase @ sky_real) + (
                        cos_phase @ sky_imag
                    )  # [chunk_vis]

                    # Combine into a complex tensor
                    vis_contribution = torch.complex(
                        real_contribution, imag_contribution
                    )

                    # Add to result
                    visibilities[b, vis_start:vis_end] += vis_contribution

        # Remove batch dimension if input was not batched
        if unbatched_input:
            visibilities = visibilities.squeeze(0)

        return visibilities
