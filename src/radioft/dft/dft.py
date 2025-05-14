import torch
from .utils import cuda_dft, cuda_idft
from radioft.utils.sizes import get_optimal_chunk_sizes


class HybridPyTorchCudaDFT:
    """
    Deterministic hybrid PyTorch-CUDA DFT implementation with performance optimizations
    """
    def __init__(self, device="cuda", max_matrix_size_gb=1.0, benchmark=True):
        self.device = device
        self.max_matrix_size_gb = max_matrix_size_gb

        # Set deterministic algorithms for more consistent performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = benchmark

        self.dft = cuda_dft
        self.idft = cuda_idft

    def forward(self, sky_values, l_coords, m_coords, n_coords, u_coords, v_coords, w_coords):
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
        vis_chunk_size, pixel_chunk_size = self.get_optimal_chunk_sizes(num_pixels, num_vis, self.max_matrix_size_gb)

        # Pre-allocate output with zeros
        visibilities = torch.zeros((batch_size, num_vis), dtype=torch.complex128, device=self.device)

        # Calculate total chunks for progress bar
        total_chunks = ((num_vis + vis_chunk_size - 1) // vis_chunk_size) * \
                      ((num_pixels + pixel_chunk_size - 1) // pixel_chunk_size)

        # Use fixed seed for more deterministic behavior
        torch.manual_seed(0)

        # Process visibility points in chunks
        for vis_start in range(0, num_vis, vis_chunk_size):
            vis_end = min(vis_start + vis_chunk_size, num_vis)
            vis_chunk_len = vis_end - vis_start

            # Get current visibility chunk
            u_chunk = u_coords[vis_start:vis_end].contiguous()  # Force contiguous for better performance
            v_chunk = v_coords[vis_start:vis_end].contiguous()
            w_chunk = w_coords[vis_start:vis_end].contiguous()

            # Pre-allocate accumulators for this visibility chunk - more efficient
            chunk_vis = torch.zeros((batch_size, vis_chunk_len),
                                    dtype=torch.complex128, device=self.device)

            # Process pixels in chunks
            for pixel_start in range(0, num_pixels, pixel_chunk_size):
                pixel_end = min(pixel_start + pixel_chunk_size, num_pixels)
                pixel_chunk_len = pixel_end - pixel_start

                # Get current pixel chunk - force contiguous for better performance
                l_chunk = l_coords[pixel_start:pixel_end].contiguous()
                m_chunk = m_coords[pixel_start:pixel_end].contiguous()
                n_chunk = n_coords[pixel_start:pixel_end].contiguous()

                sky_values_chunk = sky_values[:, pixel_start:pixel_end]
                chunk_vis += self.dft(sky_values_chunk, l_chunk, m_chunk, n_chunk, u_chunk, v_chunk, w_chunk)
            # Store result for this visibility chunk
            visibilities[:, vis_start:vis_end] = chunk_vis

            # Clean up
            del chunk_vis

        # Remove batch dimension if input was not batched
        if unbatched_input:
            visibilities = visibilities.squeeze(0)

        return visibilities

    def inverse(self, visibilities, l_coords, m_coords, n_coords, u_coords, v_coords, w_coords):
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
        vis_chunk_size, pixel_chunk_size = self._get_optimal_chunk_sizes(num_pixels, num_vis)

        # Pre-allocate output tensor for sky image
        sky_values = torch.zeros((batch_size, num_pixels), dtype=torch.complex128, device=self.device)

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
            sky_values_chunk = torch.zeros((batch_size, pixel_chunk_len),
                    dtype=torch.complex128, device=self.device)

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

                sky_values_chunk += self.idft(vis_chunk, l_chunk, m_chunk, n_chunk, u_chunk, v_chunk, w_chunk)

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
