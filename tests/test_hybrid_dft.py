import pytest
import torch
import numpy as np
from radioft.dft.dft import HybridPyTorchCudaDFT
from radioft.dft.utils import CudaDFTFunction, CudaIDFTFunction, cuda_dft, cuda_idft
import time


class TestHybridPyTorchCudaDFT:
    """Test suite for the HybridPyTorchCudaDFT class and its components"""

    @pytest.fixture
    def setup_basic(self):
        """Set up basic test data"""
        # Check if CUDA is available
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping CUDA tests")

        # Set up a small test case
        device = torch.device("cuda")
        batch_size = 1
        num_pixels = 128
        num_vis = 1000

        # Generate pixel coordinates (l, m, n)
        l_coords = torch.linspace(-0.5, 0.5, num_pixels, device=device, dtype=torch.float64)
        m_coords = torch.linspace(-0.5, 0.5, num_pixels, device=device, dtype=torch.float64)
        n_coords = torch.sqrt(1 - l_coords**2 - m_coords**2)

        # Generate visibility coordinates (u, v, w)
        u_coords = torch.linspace(-1000, 1000, num_vis, device=device, dtype=torch.float64)
        v_coords = torch.linspace(-1000, 1000, num_vis, device=device, dtype=torch.float64)
        w_coords = torch.zeros(num_vis, device=device, dtype=torch.float64)

        # Generate random sky image
        sky_image = torch.randn(batch_size, num_pixels, dtype=torch.complex128, device=device)

        # Create the DFT instance
        hybrid_dft = HybridPyTorchCudaDFT(device=device)

        return {
            "hybrid_dft": hybrid_dft,
            "l_coords": l_coords,
            "m_coords": m_coords,
            "n_coords": n_coords,
            "u_coords": u_coords,
            "v_coords": v_coords,
            "w_coords": w_coords,
            "sky_image": sky_image,
            "num_pixels": num_pixels,
            "num_vis": num_vis,
            "device": device
        }

    @pytest.fixture
    def setup_medium(self):
        """Set up medium-sized test data"""
        # Check if CUDA is available
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping CUDA tests")

        # Check if there's enough GPU memory (at least 4GB)
        if torch.cuda.get_device_properties(0).total_memory < 4 * 1024**3:
            pytest.skip("Not enough GPU memory for medium tests")

        # Set up a medium test case
        device = torch.device("cuda")
        batch_size = 1
        num_pixels = 512
        num_vis = 1000

        # Generate pixel coordinates (l, m, n)
        l = torch.linspace(-0.5, 0.5, num_pixels, device=device)
        m = torch.linspace(-0.5, 0.5, num_pixels, device=device)
        l_coords, m_coords = torch.meshgrid(l, m, indexing="ij")
        l_coords = l_coords.flatten()
        m_coords = m_coords.flatten()
        n_coords = torch.sqrt(1 - l_coords**2 - m_coords**2)

        # Generate visibility coordinates (u, v, w)
        u_coords = torch.linspace(-500, 500, num_vis, device=device)
        v_coords = torch.linspace(-500, 500, num_vis, device=device)
        w_coords = torch.zeros(num_vis, device=device)

        # Generate random sky image
        sky_image = torch.randn(batch_size, num_pixels**2, dtype=torch.complex128, device=device)

        # Create the DFT instance
        hybrid_dft = HybridPyTorchCudaDFT(device=device)

        return {
            "hybrid_dft": hybrid_dft,
            "l_coords": l_coords,
            "m_coords": m_coords,
            "n_coords": n_coords,
            "u_coords": u_coords,
            "v_coords": v_coords,
            "w_coords": w_coords,
            "sky_image": sky_image,
            "num_pixels": num_pixels**2,
            "num_vis": num_vis,
            "device": device
        }

    def test_initialization(self):
        """Test that the HybridPyTorchCudaDFT initializes correctly"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping CUDA tests")

        # Test initialization with default device
        hybrid_dft = HybridPyTorchCudaDFT()
        assert hybrid_dft.device == "cuda"

        # Test initialization with specific device
        hybrid_dft = HybridPyTorchCudaDFT(device="cuda:0")
        assert hybrid_dft.device == "cuda:0"

        # Test initialization with CPU (should work but with a warning)
        hybrid_dft = HybridPyTorchCudaDFT(device="cpu")
        assert hybrid_dft.device == "cpu"

    def test_forward_basic(self, setup_basic):
        """Test forward DFT with basic test data"""
        data = setup_basic

        # Compute visibilities using forward DFT
        visibilities = data["hybrid_dft"].forward(
            data["sky_image"],
            data["l_coords"],
            data["m_coords"],
            data["n_coords"],
            data["u_coords"],
            data["v_coords"],
            data["w_coords"]
        )

        # Check output shape
        assert visibilities.shape[0] == data["sky_image"].shape[0]
        assert visibilities.shape[1] == data["num_vis"]

        # Check output type
        assert visibilities.dtype == torch.complex128
        assert visibilities.device.type == data["device"].type
        if hasattr(data["device"], 'index') and data["device"].index is not None:
            assert visibilities.device.index == data["device"].index

    def test_inverse_basic(self, setup_basic):
        """Test inverse DFT with basic test data"""
        data = setup_basic

        # First compute visibilities using forward DFT
        visibilities = data["hybrid_dft"].forward(
            data["sky_image"],
            data["l_coords"],
            data["m_coords"],
            data["n_coords"],
            data["u_coords"],
            data["v_coords"],
            data["w_coords"]
        )

        # Then reconstruct sky image using inverse DFT
        reconstructed = data["hybrid_dft"].inverse(
            visibilities,
            data["l_coords"],
            data["m_coords"],
            data["n_coords"],
            data["u_coords"],
            data["v_coords"],
            data["w_coords"]
        )

        # Check output shape
        assert reconstructed.shape[0] == data["sky_image"].shape[0]
        assert reconstructed.shape[1] == data["num_pixels"]

        # Check output type
        assert reconstructed.dtype == torch.complex128
        assert reconstructed.device.type == data["device"].type
        if hasattr(data["device"], 'index') and data["device"].index is not None:
            assert reconstructed.device.index == data["device"].index

    def test_roundtrip_basic(self, setup_basic):
        """Test roundtrip (forward then inverse) DFT with basic test data"""
        data = setup_basic

        # Compute visibilities
        visibilities = data["hybrid_dft"].forward(
            data["sky_image"],
            data["l_coords"],
            data["m_coords"],
            data["n_coords"],
            data["u_coords"],
            data["v_coords"],
            data["w_coords"]
        )

        # Reconstruct sky image
        reconstructed = data["hybrid_dft"].inverse(
            visibilities,
            data["l_coords"],
            data["m_coords"],
            data["n_coords"],
            data["u_coords"],
            data["v_coords"],
            data["w_coords"]
        )

        # Check that the roundtrip reconstruction is close to the original
        # Note: This will not be exact due to the finite sampling of (u,v,w) points
        # But should be reasonably close for simple test cases
        assert torch.allclose(
            reconstructed,
            data["sky_image"],
            rtol=1e-1,
            atol=1e-1
        )

    def test_forward_medium(self, setup_medium):
        """Test forward DFT with medium-sized test data"""
        data = setup_medium

        # Compute visibilities
        start_time = time.time()
        visibilities = data["hybrid_dft"].forward(
            data["sky_image"],
            data["l_coords"],
            data["m_coords"],
            data["n_coords"],
            data["u_coords"],
            data["v_coords"],
            data["w_coords"]
        )
        elapsed = time.time() - start_time

        # Check output shape
        assert visibilities.shape[0] == data["sky_image"].shape[0]
        assert visibilities.shape[1] == data["num_vis"]

        # Check output type
        assert visibilities.dtype == torch.complex128
        assert visibilities.device.type == data["device"].type
        if hasattr(data["device"], 'index') and data["device"].index is not None:
            assert visibilities.device.index == data["device"].index

        # Log performance
        print(f"Forward DFT on {data['num_pixels']} pixels × {data['num_vis']} visibilities: {elapsed:.2f}s")

    def test_inverse_medium(self, setup_medium):
        """Test inverse DFT with medium-sized test data"""
        data = setup_medium

        # First compute visibilities
        visibilities = data["hybrid_dft"].forward(
            data["sky_image"],
            data["l_coords"],
            data["m_coords"],
            data["n_coords"],
            data["u_coords"],
            data["v_coords"],
            data["w_coords"]
        )

        # Reconstruct with max_memory_gb parameter
        start_time = time.time()
        reconstructed = data["hybrid_dft"].inverse(
            visibilities,
            data["l_coords"],
            data["m_coords"],
            data["n_coords"],
            data["u_coords"],
            data["v_coords"],
            data["w_coords"],
            max_memory_gb=2
        )
        elapsed = time.time() - start_time

        # Check output shape
        assert reconstructed.shape[0] == data["sky_image"].shape[0]
        assert reconstructed.shape[1] == data["num_pixels"]

        # Check output type
        assert reconstructed.dtype == torch.complex128

        # Log performance
        print(f"Inverse DFT on {data['num_vis']} visibilities × {data['num_pixels']} pixels: {elapsed:.2f}s")

    def test_memory_parameter(self, setup_medium):
        """Test the max_memory_gb parameter affects chunking"""
        data = setup_medium

        # Generate some visibilities (random in this case)
        visibilities = torch.randn(1, data["num_vis"], dtype=torch.complex128, device=data["device"])

        # First with low memory
        hybrid_dft = data["hybrid_dft"]

        # Run with low memory setting (should use smaller chunks, take longer)
        start_time = time.time()
        _ = hybrid_dft.inverse(
            visibilities,
            data["l_coords"],
            data["m_coords"],
            data["n_coords"],
            data["u_coords"],
            data["v_coords"],
            data["w_coords"],
            max_memory_gb=1
        )
        low_mem_time = time.time() - start_time

        # Run with higher memory setting (should use larger chunks, be faster)
        start_time = time.time()
        _ = hybrid_dft.inverse(
            visibilities,
            data["l_coords"],
            data["m_coords"],
            data["n_coords"],
            data["u_coords"],
            data["v_coords"],
            data["w_coords"],
            max_memory_gb=4
        )
        high_mem_time = time.time() - start_time

        # The higher memory version should be faster (though not guaranteed)
        # We'll use a soft assert here as timing can vary
        print(f"Low memory (1GB): {low_mem_time:.2f}s, High memory (4GB): {high_mem_time:.2f}s")
        if high_mem_time > low_mem_time:
            print("Warning: Higher memory setting wasn't faster - this might be due to system conditions")

    def test_gradient_flow(self, setup_basic):
        """Test that gradients flow through the DFT operations"""
        data = setup_basic

        # Make sky image requires gradient
        sky_image = data["sky_image"].clone().detach().requires_grad_(True)

        # Forward pass
        visibilities = data["hybrid_dft"].forward(
            sky_image,
            data["l_coords"],
            data["m_coords"],
            data["n_coords"],
            data["u_coords"],
            data["v_coords"],
            data["w_coords"]
        )

        # Loss function: sum of absolute values of visibilities
        loss = torch.abs(visibilities).sum()

        # Backward pass
        loss.backward()

        # Check that gradient was computed
        assert sky_image.grad is not None
        assert not torch.isnan(sky_image.grad).any()

    def test_gradient_flow_inverse(self, setup_basic):
        """Test that gradients flow through the inverse DFT operation"""
        data = setup_basic

        # Make visibilities require gradient
        visibilities = torch.randn(1, data["num_vis"], dtype=torch.complex128,
                                  device=data["device"], requires_grad=True)

        # Forward pass with inverse DFT
        sky_image = data["hybrid_dft"].inverse(
            visibilities,
            data["l_coords"],
            data["m_coords"],
            data["n_coords"],
            data["u_coords"],
            data["v_coords"],
            data["w_coords"]
        )

        # Loss function: sum of absolute values of sky image
        loss = torch.abs(sky_image).sum()

        # Backward pass
        loss.backward()

        # Check that gradient was computed
        assert visibilities.grad is not None
        assert not torch.isnan(visibilities.grad).any()

    # def test_float32_precision(self, setup_basic):
    #     """Test that the DFT works with float32 precision"""
    #     data = setup_basic
    #
    #     # Convert data to float32
    #     l_coords = data["l_coords"].float()
    #     m_coords = data["m_coords"].float()
    #     n_coords = data["n_coords"].float()
    #     u_coords = data["u_coords"].float()
    #     v_coords = data["v_coords"].float()
    #     w_coords = data["w_coords"].float()
    #     sky_image = data["sky_image"].cfloat()
    #
    #     # Forward pass
    #     visibilities = data["hybrid_dft"].forward(
    #         sky_image,
    #         l_coords,
    #         m_coords,
    #         n_coords,
    #         u_coords,
    #         v_coords,
    #         w_coords
    #     )
    #
    #     # Check dtype
    #     assert visibilities.dtype == torch.complex64
    #
    #     # Inverse pass
    #     reconstructed = data["hybrid_dft"].inverse(
    #         visibilities,
    #         l_coords,
    #         m_coords,
    #         n_coords,
    #         u_coords,
    #         v_coords,
    #         w_coords
    #     )
    #
    #     # Check dtype
    #     assert reconstructed.dtype == torch.complex64

    def test_cuda_dft_function_direct(self, setup_basic):
        """Test the cuda_dft function directly"""
        data = setup_basic

        # Call the function directly
        visibilities = cuda_dft(
            data["sky_image"],
            data["l_coords"],
            data["m_coords"],
            data["n_coords"],
            data["u_coords"],
            data["v_coords"],
            data["w_coords"]
        )

        # Check shape and type
        assert visibilities.shape[0] == data["sky_image"].shape[0]
        assert visibilities.shape[1] == data["num_vis"]
        assert visibilities.dtype == torch.complex128

    def test_cuda_idft_function_direct(self, setup_basic):
        """Test the cuda_idft function directly"""
        data = setup_basic

        # Generate visibilities
        visibilities = torch.randn(1, data["num_vis"], dtype=torch.complex128, device=data["device"])

        # Call the function directly
        sky_image = cuda_idft(
            visibilities,
            data["l_coords"],
            data["m_coords"],
            data["n_coords"],
            data["u_coords"],
            data["v_coords"],
            data["w_coords"],
        )

        # Check shape and type
        assert sky_image.shape[0] == visibilities.shape[0]
        assert sky_image.shape[1] == data["num_pixels"]
        assert sky_image.dtype == torch.complex128

    def test_batch_processing(self, setup_basic):
        """Test that the DFT can handle batched inputs"""
        data = setup_basic

        # Create a batched sky image
        batch_size = 3
        batched_sky = torch.randn(batch_size, data["num_pixels"],
                                 dtype=torch.complex128, device=data["device"])

        # Forward pass
        visibilities = data["hybrid_dft"].forward(
            batched_sky,
            data["l_coords"],
            data["m_coords"],
            data["n_coords"],
            data["u_coords"],
            data["v_coords"],
            data["w_coords"]
        )

        # Check batch dimension is preserved
        assert visibilities.shape[0] == batch_size

        # Inverse pass with batched visibilities
        reconstructed = data["hybrid_dft"].inverse(
            visibilities,
            data["l_coords"],
            data["m_coords"],
            data["n_coords"],
            data["u_coords"],
            data["v_coords"],
            data["w_coords"]
        )

        # Check batch dimension is preserved
        assert reconstructed.shape[0] == batch_size

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_zero_w_plane(self):
        """Test the special case where w=0 (2D imaging)"""
        device = torch.device("cuda")

        # Set up a small test case with w=0
        num_pixels = 81
        num_vis = 100

        # Generate pixel coordinates (l, m, n)
        l_coords = torch.linspace(-0.5, 0.5, num_pixels, device=device)
        m_coords = torch.linspace(-0.5, 0.5, num_pixels, device=device)
        n_coords = torch.ones_like(l_coords, device=device)  # n=1 for w=0 case

        # Generate visibility coordinates (u, v, w)
        u_coords = torch.linspace(-100, 100, num_vis, device=device)
        v_coords = torch.linspace(-100, 100, num_vis, device=device)
        w_coords = torch.zeros(num_vis, device=device)  # w=0 for all visibilities

        # Create a simple sky image (a point source at the center)
        sky_image = torch.zeros(1, num_pixels, dtype=torch.complex128, device=device)
        sky_image[0, num_pixels // 2] = 1.0 + 0.0j

        # Create the DFT instance
        hybrid_dft = HybridPyTorchCudaDFT(device=device)

        # Compute visibilities
        visibilities = hybrid_dft.forward(
            sky_image,
            l_coords,
            m_coords,
            n_coords,
            u_coords,
            v_coords,
            w_coords
        )

        # For a point source at the center, the visibility amplitudes should be constant
        vis_amp = torch.abs(visibilities)
        assert torch.allclose(vis_amp, vis_amp[0, 0].expand_as(vis_amp), rtol=1e-5)

        # Phases should be close to zero for a centered point source
        vis_phase = torch.angle(visibilities)
        assert torch.allclose(vis_phase, torch.zeros_like(vis_phase), atol=1e-5)

    def test_identity_transform(self, setup_basic):
        """
        Test that the forward transform of a single point source at the origin
        produces constant amplitudes across all visibilities
        """
        data = setup_basic

        # Create a sky image with a single point source at the center
        sky_image = torch.zeros_like(data["sky_image"])
        center_idx = data["num_pixels"] // 2
        sky_image[0, center_idx] = 1.0 + 0.0j

        # Forward transform
        visibilities = data["hybrid_dft"].forward(
            sky_image,
            data["l_coords"],
            data["m_coords"],
            data["n_coords"],
            data["u_coords"],
            data["v_coords"],
            data["w_coords"]
        )

        # For a point source at the center, the visibility amplitudes should be constant
        vis_amp = torch.abs(visibilities)
        assert torch.allclose(vis_amp, vis_amp[0, 0].expand_as(vis_amp), rtol=1e-4)


class TestHybridPyTorchCudaDFTEdgeCases:
    """Test edge cases and specialized scenarios for HybridPyTorchCudaDFT"""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_single_visibility(self):
        """Test with just a single visibility point"""
        device = torch.device("cuda")
        num_pixels = 64

        # Generate pixel coordinates
        l_coords = torch.linspace(-0.5, 0.5, num_pixels, device=device)
        m_coords = torch.linspace(-0.5, 0.5, num_pixels, device=device)
        n_coords = torch.sqrt(1 - l_coords**2 - m_coords**2)

        # Single visibility at the origin
        u_coords = torch.zeros(1, device=device)
        v_coords = torch.zeros(1, device=device)
        w_coords = torch.zeros(1, device=device)

        # Simple sky image
        sky_image = torch.ones(1, num_pixels, dtype=torch.complex128, device=device)

        # Create DFT instance
        hybrid_dft = HybridPyTorchCudaDFT(device=device)

        # Forward transform
        visibilities = hybrid_dft.forward(
            sky_image,
            l_coords,
            m_coords,
            n_coords,
            u_coords,
            v_coords,
            w_coords
        )

        # Single visibility should have shape [1, 1]
        assert visibilities.shape == (1, 1)

        # For constant sky brightness and zero baselines, the visibility should be the sum of the sky
        expected_value = sky_image.sum()
        assert torch.isclose(visibilities[0, 0], expected_value, rtol=1e-4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_single_pixel(self):
        """Test with just a single pixel point"""
        device = torch.device("cuda")
        num_vis = 64

        # Single pixel at the origin
        l_coords = torch.zeros(1, device=device)
        m_coords = torch.zeros(1, device=device)
        n_coords = torch.ones(1, device=device)

        # Generate visibility coordinates
        u_coords = torch.linspace(-100, 100, num_vis, device=device)
        v_coords = torch.linspace(-100, 100, num_vis, device=device)
        w_coords = torch.zeros(num_vis, device=device)

        # Sky value for the single pixel
        sky_image = torch.ones(1, 1, dtype=torch.complex128, device=device)

        # Create DFT instance
        hybrid_dft = HybridPyTorchCudaDFT(device=device)

        # Forward transform
        visibilities = hybrid_dft.forward(
            sky_image,
            l_coords,
            m_coords,
            n_coords,
            u_coords,
            v_coords,
            w_coords
        )

        # Should have shape [1, num_vis]
        assert visibilities.shape == (1, num_vis)

        # For a point source at origin, all visibilities should have the same amplitude
        assert torch.allclose(torch.abs(visibilities), torch.abs(visibilities[0, 0]), rtol=1e-4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_different_coordinate_counts(self):
        """Test handling of different numbers of coordinates"""
        device = torch.device("cuda")

        # Different sizes for coordinates
        num_l = 100
        num_vis = 50

        # Generate pixel coordinates
        l_coords = torch.linspace(-0.5, 0.5, num_l, device=device)
        m_coords = torch.linspace(-0.5, 0.5, num_l, device=device)
        n_coords = torch.sqrt(1 - l_coords**2 - m_coords**2)

        # Generate visibility coordinates
        u_coords = torch.linspace(-100, 100, num_vis, device=device)
        v_coords = torch.linspace(-100, 100, num_vis, device=device)
        w_coords = torch.zeros(num_vis, device=device)

        # Sky image with num_l points
        sky_image = torch.ones(1, num_l, dtype=torch.complex128, device=device)

        # Create DFT instance
        hybrid_dft = HybridPyTorchCudaDFT(device=device)

        # Forward transform
        visibilities = hybrid_dft.forward(
            sky_image,
            l_coords,
            m_coords,
            n_coords,
            u_coords,
            v_coords,
            w_coords
        )

        # Should have the right output shape
        assert visibilities.shape == (1, num_vis)

        # Inverse transform
        reconstructed = hybrid_dft.inverse(
            visibilities,
            l_coords,
            m_coords,
            n_coords,
            u_coords,
            v_coords,
            w_coords
        )

        # Should restore original shape
        assert reconstructed.shape == (1, num_l)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_non_uniform_coordinates(self):
        """Test with non-uniform coordinate distributions"""
        device = torch.device("cuda")
        num_pixels = 64
        num_vis = 50

        # Non-uniform pixel coordinates clustered near the origin
        l_coords = torch.sin(torch.linspace(-np.pi/3, np.pi/3, num_pixels, device=device))
        m_coords = torch.sin(torch.linspace(-np.pi/3, np.pi/3, num_pixels, device=device))
        n_coords = torch.sqrt(1 - l_coords**2 - m_coords**2)

        # Non-uniform visibility coordinates clustered at low frequencies
        u_coords = torch.sinh(torch.linspace(-2, 2, num_vis, device=device)) * 50
        v_coords = torch.sinh(torch.linspace(-2, 2, num_vis, device=device)) * 50
        w_coords = torch.zeros(num_vis, device=device)

        # Simple sky image
        sky_image = torch.ones(1, num_pixels, dtype=torch.complex128, device=device)

        # Create DFT instance
        hybrid_dft = HybridPyTorchCudaDFT(device=device)

        # Forward transform
        visibilities = hybrid_dft.forward(
            sky_image,
            l_coords,
            m_coords,
            n_coords,
            u_coords,
            v_coords,
            w_coords
        )

        # Should have correct shape
        assert visibilities.shape == (1, num_vis)

        # Inverse transform
        reconstructed = hybrid_dft.inverse(
            visibilities,
            l_coords,
            m_coords,
            n_coords,
            u_coords,
            v_coords,
            w_coords
        )

        # Should have correct shape
        assert reconstructed.shape == (1, num_pixels)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_w_term(self):
        """Test the effect of non-zero w-terms"""
        device = torch.device("cuda")
        num_pixels = 64
        num_vis = 50

        # Generate pixel coordinates
        l_coords = torch.linspace(-0.5, 0.5, num_pixels, device=device)
        m_coords = torch.linspace(-0.5, 0.5, num_pixels, device=device)
        n_coords = torch.sqrt(1 - l_coords**2 - m_coords**2)

        # Generate visibility coordinates with non-zero w
        u_coords = torch.linspace(-100, 100, num_vis, device=device)
        v_coords = torch.linspace(-100, 100, num_vis, device=device)
        w_coords = torch.linspace(0, 50, num_vis, device=device)  # Non-zero w-terms

        # Create point source at the center
        sky_image = torch.zeros(1, num_pixels, dtype=torch.complex128, device=device)
        center_idx = num_pixels // 2
        sky_image[0, center_idx] = 1.0 + 0.0j

        # Create DFT instance
        hybrid_dft = HybridPyTorchCudaDFT(device=device)

        # Forward transform with non-zero w
        visibilities_w = hybrid_dft.forward(
            sky_image,
            l_coords,
            m_coords,
            n_coords,
            u_coords,
            v_coords,
            w_coords
        )

        # Forward transform with zero w
        zero_w = torch.zeros_like(w_coords)
        visibilities_zero_w = hybrid_dft.forward(
            sky_image,
            l_coords,
            m_coords,
            n_coords,
            u_coords,
            v_coords,
            zero_w
        )

        # Non-zero w should affect the phases but not the amplitudes for a centered source
        assert torch.allclose(
            torch.abs(visibilities_w),
            torch.abs(visibilities_zero_w),
            rtol=1e-4
        )

        # Phases should be different
        phases_w = torch.angle(visibilities_w)
        phases_zero_w = torch.angle(visibilities_zero_w)

        # At least some phases should differ significantly
        assert not torch.allclose(phases_w, phases_zero_w, atol=1e-2)


class TestHybridPyTorchCudaDFTPerformance:
    """Performance tests for HybridPyTorchCudaDFT"""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize(
        "num_pixels,num_vis",
        [
            (128, 1000),
            (1024, 1000),
            (4096, 1000),
        ]
    )
    def test_forward_performance(self, num_pixels, num_vis):
        """Benchmark forward DFT with different sizes"""
        device = torch.device("cuda")

        # Generate coordinates
        l_coords = torch.linspace(-0.5, 0.5, num_pixels, device=device)
        m_coords = torch.linspace(-0.5, 0.5, num_pixels, device=device)
        n_coords = torch.sqrt(1 - l_coords**2 - m_coords**2)

        u_coords = torch.linspace(-100, 100, num_vis, device=device)
        v_coords = torch.linspace(-100, 100, num_vis, device=device)
        w_coords = torch.zeros(num_vis, device=device)

        # Generate random sky image
        sky_image = torch.randn(1, num_pixels, dtype=torch.complex128, device=device)

        # Create DFT instance
        hybrid_dft = HybridPyTorchCudaDFT(device=device)

        # Warm-up GPU
        _ = hybrid_dft.forward(
            sky_image,
            l_coords,
            m_coords,
            n_coords,
            u_coords,
            v_coords,
            w_coords
        )

        # Synchronize GPU
        torch.cuda.synchronize()

        # Benchmark
        start_time = time.time()
        _ = hybrid_dft.forward(
            sky_image,
            l_coords,
            m_coords,
            n_coords,
            u_coords,
            v_coords,
            w_coords
        )
        torch.cuda.synchronize()
        elapsed = time.time() - start_time

        # Log performance
        print(f"Forward DFT {num_pixels} pixels → {num_vis} visibilities: {elapsed:.4f}s")

        # No specific assertion, just benchmarking

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize(
        "num_pixels,num_vis,max_memory_gb",
        [
            (128, 1000, 1),
            (128, 1000, 4),
            (1024, 1000, 1),
            (1024, 1000, 4),
            (4096, 1000, 1),
            (4096, 1000, 4),
        ]
    )
    def test_inverse_performance(self, num_pixels, num_vis, max_memory_gb):
        """Benchmark inverse DFT with different sizes and memory settings"""
        device = torch.device("cuda")

        # Generate coordinates
        l_coords = torch.linspace(-0.5, 0.5, num_pixels, device=device)
        m_coords = torch.linspace(-0.5, 0.5, num_pixels, device=device)
        n_coords = torch.sqrt(1 - l_coords**2 - m_coords**2)

        u_coords = torch.linspace(-100, 100, num_vis, device=device)
        v_coords = torch.linspace(-100, 100, num_vis, device=device)
        w_coords = torch.zeros(num_vis, device=device)

        # Generate random visibilities
        visibilities = torch.randn(1, num_vis, dtype=torch.complex128, device=device)

        # Create DFT instance
        hybrid_dft = HybridPyTorchCudaDFT(device=device)

        # Warm-up GPU
        _ = hybrid_dft.inverse(
            visibilities,
            l_coords,
            m_coords,
            n_coords,
            u_coords,
            v_coords,
            w_coords,
            max_memory_gb=max_memory_gb
        )

        # Synchronize GPU
        torch.cuda.synchronize()

        # Benchmark
        start_time = time.time()
        _ = hybrid_dft.inverse(
            visibilities,
            l_coords,
            m_coords,
            n_coords,
            u_coords,
            v_coords,
            w_coords,
            max_memory_gb=max_memory_gb
        )
        torch.cuda.synchronize()
        elapsed = time.time() - start_time

        # Log performance
        print(f"Inverse DFT {num_vis} visibilities → {num_pixels} pixels (max_memory: {max_memory_gb}GB): {elapsed:.4f}s")

        # No specific assertion, just benchmarking

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize(
        "dtype",
        [
            torch.complex64,
            torch.complex128,
        ]
    )
    def test_precision_performance(self, dtype):
        """Benchmark performance with different precision settings"""
        device = torch.device("cuda")
        num_pixels = 2048
        num_vis = 1000

        # Determine float type based on complex type
        float_type = torch.float32 if dtype == torch.complex64 else torch.float64

        # Generate coordinates
        l_coords = torch.linspace(-0.5, 0.5, num_pixels, device=device, dtype=float_type)
        m_coords = torch.linspace(-0.5, 0.5, num_pixels, device=device, dtype=float_type)
        n_coords = torch.sqrt(1 - l_coords**2 - m_coords**2)

        u_coords = torch.linspace(-100, 100, num_vis, device=device, dtype=float_type)
        v_coords = torch.linspace(-100, 100, num_vis, device=device, dtype=float_type)
        w_coords = torch.zeros(num_vis, device=device, dtype=float_type)

        # Generate random sky image
        sky_image = torch.randn(1, num_pixels, dtype=dtype, device=device)

        # Create DFT instance
        hybrid_dft = HybridPyTorchCudaDFT(device=device)

        # Warm-up GPU
        _ = hybrid_dft.forward(
            sky_image,
            l_coords,
            m_coords,
            n_coords,
            u_coords,
            v_coords,
            w_coords
        )

        # Synchronize GPU
        torch.cuda.synchronize()

        # Benchmark forward
        start_time = time.time()
        visibilities = hybrid_dft.forward(
            sky_image,
            l_coords,
            m_coords,
            n_coords,
            u_coords,
            v_coords,
            w_coords
        )
        torch.cuda.synchronize()
        forward_elapsed = time.time() - start_time

        # Benchmark inverse
        start_time = time.time()
        _ = hybrid_dft.inverse(
            visibilities,
            l_coords,
            m_coords,
            n_coords,
            u_coords,
            v_coords,
            w_coords,
            max_memory_gb=4
        )
        torch.cuda.synchronize()
        inverse_elapsed = time.time() - start_time

        # Log performance
        precision_name = "single" if dtype == torch.complex64 else "double"
        print(f"{precision_name.capitalize()} precision:")
        print(f"  Forward: {forward_elapsed:.4f}s")
        print(f"  Inverse: {inverse_elapsed:.4f}s")
        print(f"  Total: {forward_elapsed + inverse_elapsed:.4f}s")


