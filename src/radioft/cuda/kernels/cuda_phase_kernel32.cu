#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// Constants for numerical stability
#define PI 3.1415927

/**
 * Phase calculation kernel
 * This kernel computes the phase matrix with float32 precision
 */
__global__ void compute_phase_kernel32(
    const float* __restrict__ l_coords, const float* __restrict__ m_coords,
    const float* __restrict__ n_coords, const float* __restrict__ u_coords,
    const float* __restrict__ v_coords, const float* __restrict__ w_coords,
    float* __restrict__ phase_matrix, int num_vis, int num_pixels) {
  // Each thread computes one element of the phase matrix
  int vis_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int pixel_idx = blockIdx.y * blockDim.y + threadIdx.y;

  if (vis_idx < num_vis && pixel_idx < num_pixels) {
    // Load coordinates with float32 precision
    float u = u_coords[vis_idx];
    float v = v_coords[vis_idx];
    float w = w_coords[vis_idx];
    float l = l_coords[pixel_idx];
    float m = m_coords[pixel_idx];
    float n = n_coords[pixel_idx] - 1.0;  // n - 1

    // Careful phase computation to maintain precision
    // Calculate components separately to minimize rounding errors
    float ul = u * l;
    float vm = v * m;
    float wn = w * n;

    // Combine terms carefully
    float sum1 = ul + vm;
    float sum2 = sum1 + wn;

    // Final phase calculation
    float phase = -2.0 * PI * sum2;

    // Store phase in matrix
    phase_matrix[vis_idx * num_pixels + pixel_idx] = phase;
  }
}

// Wrapper functions for PyTorch integration

torch::Tensor compute_phase_matrix32(
    torch::Tensor l_coords, torch::Tensor m_coords, torch::Tensor n_coords,
    torch::Tensor u_coords, torch::Tensor v_coords, torch::Tensor w_coords) {
  // Validate inputs
  TORCH_CHECK(l_coords.is_cuda(), "l_coords must be a CUDA tensor");
  TORCH_CHECK(m_coords.is_cuda(), "m_coords must be a CUDA tensor");
  TORCH_CHECK(n_coords.is_cuda(), "n_coords must be a CUDA tensor");
  TORCH_CHECK(u_coords.is_cuda(), "u_coords must be a CUDA tensor");
  TORCH_CHECK(v_coords.is_cuda(), "v_coords must be a CUDA tensor");
  TORCH_CHECK(w_coords.is_cuda(), "w_coords must be a CUDA tensor");

  int num_pixels = l_coords.size(0);
  int num_vis = u_coords.size(0);

  // Create output tensor
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(l_coords.device());
  auto phase_matrix = torch::empty({num_vis, num_pixels}, options);

  // Set up kernel execution
  dim3 threads_per_block(16, 16);
  dim3 num_blocks((num_vis + threads_per_block.x - 1) / threads_per_block.x,
                  (num_pixels + threads_per_block.y - 1) / threads_per_block.y);

  // Launch kernel
  compute_phase_kernel32<<<num_blocks, threads_per_block>>>(
      l_coords.data_ptr<float>(), m_coords.data_ptr<float>(),
      n_coords.data_ptr<float>(), u_coords.data_ptr<float>(),
      v_coords.data_ptr<float>(), w_coords.data_ptr<float>(),
      phase_matrix.data_ptr<float>(), num_vis, num_pixels);

  return phase_matrix;
}

/**
 * Inverse phase calculation kernel
 * This kernel computes the phase matrix with float32
 * and POSITIVE sign for inverse DFT
 */
__global__ void compute_inverse_phase_kernel32(
    const float* __restrict__ l_coords, const float* __restrict__ m_coords,
    const float* __restrict__ n_coords, const float* __restrict__ u_coords,
    const float* __restrict__ v_coords, const float* __restrict__ w_coords,
    float* __restrict__ phase_matrix, int num_vis, int num_pixels) {
  // Each thread computes one element of the phase matrix
  int vis_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int pixel_idx = blockIdx.y * blockDim.y + threadIdx.y;

  if (vis_idx < num_vis && pixel_idx < num_pixels) {
    // Load coordinates with float32 precision
    float u = u_coords[vis_idx];
    float v = v_coords[vis_idx];
    float w = w_coords[vis_idx];
    float l = l_coords[pixel_idx];
    float m = m_coords[pixel_idx];
    float n = n_coords[pixel_idx] - 1.0;  // n - 1

    // Careful phase computation to maintain precision
    float ul = u * l;
    float vm = v * m;
    float wn = w * n;

    float sum1 = ul + vm;
    float sum2 = sum1 + wn;

    // Positive sign for inverse DFT (key difference)
    float phase = 2.0 * PI * sum2;

    // Store phase in matrix
    phase_matrix[vis_idx * num_pixels + pixel_idx] = phase;
  }
}

// Adding an inverse DFT version of the phase kernel

torch::Tensor compute_inverse_phase_matrix32(
    torch::Tensor l_coords, torch::Tensor m_coords, torch::Tensor n_coords,
    torch::Tensor u_coords, torch::Tensor v_coords, torch::Tensor w_coords) {
  // Input validation identical to the forward version
  TORCH_CHECK(l_coords.is_cuda(), "l_coords must be a CUDA tensor");
  TORCH_CHECK(m_coords.is_cuda(), "m_coords must be a CUDA tensor");
  TORCH_CHECK(n_coords.is_cuda(), "n_coords must be a CUDA tensor");
  TORCH_CHECK(u_coords.is_cuda(), "u_coords must be a CUDA tensor");
  TORCH_CHECK(v_coords.is_cuda(), "v_coords must be a CUDA tensor");
  TORCH_CHECK(w_coords.is_cuda(), "w_coords must be a CUDA tensor");

  int num_pixels = l_coords.size(0);
  int num_vis = u_coords.size(0);

  // Create output tensor
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(l_coords.device());
  auto phase_matrix = torch::empty({num_vis, num_pixels}, options);

  // Set up kernel execution
  dim3 threads_per_block(16, 16);
  dim3 num_blocks((num_vis + threads_per_block.x - 1) / threads_per_block.x,
                  (num_pixels + threads_per_block.y - 1) / threads_per_block.y);

  // Launch inverse kernel (with positive sign)
  compute_inverse_phase_kernel32<<<num_blocks, threads_per_block>>>(
      l_coords.data_ptr<float>(), m_coords.data_ptr<float>(),
      n_coords.data_ptr<float>(), u_coords.data_ptr<float>(),
      v_coords.data_ptr<float>(), w_coords.data_ptr<float>(),
      phase_matrix.data_ptr<float>(), num_vis, num_pixels);

  return phase_matrix;
}
