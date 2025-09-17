#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// Constants for numerical stability
#define PI_DOUBLE 3.14159265358979323846

/**
 * High-precision phase calculation kernel
 * This kernel computes the phase matrix with double precision
 */
__global__ void compute_high_precision_phase_kernel(
    const double* __restrict__ l_coords, const double* __restrict__ m_coords,
    const double* __restrict__ n_coords, const double* __restrict__ u_coords,
    const double* __restrict__ v_coords, const double* __restrict__ w_coords,
    double* __restrict__ phase_matrix, int num_vis, int num_pixels) {
  // Each thread computes one element of the phase matrix
  int vis_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int pixel_idx = blockIdx.y * blockDim.y + threadIdx.y;

  if (vis_idx < num_vis && pixel_idx < num_pixels) {
    // Load coordinates with high precision
    double u = u_coords[vis_idx];
    double v = v_coords[vis_idx];
    double w = w_coords[vis_idx];
    double l = l_coords[pixel_idx];
    double m = m_coords[pixel_idx];
    double n = n_coords[pixel_idx] - 1.0;  // n - 1

    // Careful phase computation to maintain precision
    // Calculate components separately to minimize rounding errors
    double ul = u * l;
    double vm = v * m;
    double wn = w * n;

    // Combine terms carefully
    double sum1 = ul + vm;
    double sum2 = sum1 + wn;

    // Final phase calculation
    double phase = -2.0 * PI_DOUBLE * sum2;

    // Store phase in matrix
    phase_matrix[vis_idx * num_pixels + pixel_idx] = phase;
  }
}

// Wrapper functions for PyTorch integration

torch::Tensor compute_phase_matrix(
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
      torch::TensorOptions().dtype(torch::kFloat64).device(l_coords.device());
  auto phase_matrix = torch::empty({num_vis, num_pixels}, options);

  // Set up kernel execution
  dim3 threads_per_block(16, 16);
  dim3 num_blocks((num_vis + threads_per_block.x - 1) / threads_per_block.x,
                  (num_pixels + threads_per_block.y - 1) / threads_per_block.y);

  // Launch kernel
  compute_high_precision_phase_kernel<<<num_blocks, threads_per_block>>>(
      l_coords.data_ptr<double>(), m_coords.data_ptr<double>(),
      n_coords.data_ptr<double>(), u_coords.data_ptr<double>(),
      v_coords.data_ptr<double>(), w_coords.data_ptr<double>(),
      phase_matrix.data_ptr<double>(), num_vis, num_pixels);

  return phase_matrix;
}

/**
 * High-precision inverse phase calculation kernel
 * This kernel computes the phase matrix with double precision
 * and POSITIVE sign for inverse DFT
 */
__global__ void compute_high_precision_inverse_phase_kernel(
    const double* __restrict__ l_coords, const double* __restrict__ m_coords,
    const double* __restrict__ n_coords, const double* __restrict__ u_coords,
    const double* __restrict__ v_coords, const double* __restrict__ w_coords,
    double* __restrict__ phase_matrix, int num_vis, int num_pixels) {
  // Each thread computes one element of the phase matrix
  int vis_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int pixel_idx = blockIdx.y * blockDim.y + threadIdx.y;

  if (vis_idx < num_vis && pixel_idx < num_pixels) {
    // Load coordinates with high precision
    double u = u_coords[vis_idx];
    double v = v_coords[vis_idx];
    double w = w_coords[vis_idx];
    double l = l_coords[pixel_idx];
    double m = m_coords[pixel_idx];
    double n = n_coords[pixel_idx] - 1.0;  // n - 1

    // Careful phase computation to maintain precision
    double ul = u * l;
    double vm = v * m;
    double wn = w * n;

    double sum1 = ul + vm;
    double sum2 = sum1 + wn;

    // Positive sign for inverse DFT (key difference)
    double phase = 2.0 * PI_DOUBLE * sum2;

    // Store phase in matrix
    phase_matrix[vis_idx * num_pixels + pixel_idx] = phase;
  }
}

// Adding an inverse DFT version of the phase kernel

torch::Tensor compute_inverse_phase_matrix(
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
      torch::TensorOptions().dtype(torch::kFloat64).device(l_coords.device());
  auto phase_matrix = torch::empty({num_vis, num_pixels}, options);

  // Set up kernel execution
  dim3 threads_per_block(16, 16);
  dim3 num_blocks((num_vis + threads_per_block.x - 1) / threads_per_block.x,
                  (num_pixels + threads_per_block.y - 1) / threads_per_block.y);

  // Launch inverse kernel (with positive sign)
  compute_high_precision_inverse_phase_kernel<<<num_blocks,
                                                threads_per_block>>>(
      l_coords.data_ptr<double>(), m_coords.data_ptr<double>(),
      n_coords.data_ptr<double>(), u_coords.data_ptr<double>(),
      v_coords.data_ptr<double>(), w_coords.data_ptr<double>(),
      phase_matrix.data_ptr<double>(), num_vis, num_pixels);

  return phase_matrix;
}
