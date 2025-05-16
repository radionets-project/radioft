#include <torch/extension.h>

torch::Tensor
compute_phase_matrix(torch::Tensor l_coords, torch::Tensor m_coords,
                     torch::Tensor n_coords, torch::Tensor u_coords,
                     torch::Tensor v_coords, torch::Tensor w_coords);

torch::Tensor
compute_inverse_phase_matrix(torch::Tensor l_coords, torch::Tensor m_coords,
                             torch::Tensor n_coords, torch::Tensor u_coords,
                             torch::Tensor v_coords, torch::Tensor w_coords);

torch::Tensor
compute_phase_matrix32(torch::Tensor l_coords, torch::Tensor m_coords,
                       torch::Tensor n_coords, torch::Tensor u_coords,
                       torch::Tensor v_coords, torch::Tensor w_coords);

torch::Tensor
compute_inverse_phase_matrix32(torch::Tensor l_coords, torch::Tensor m_coords,
                               torch::Tensor n_coords, torch::Tensor u_coords,
                               torch::Tensor v_coords, torch::Tensor w_coords);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // float64 implemetations
  m.def("compute_phase_matrix", &compute_phase_matrix,
        "Compute phase matrix with high precision");
  m.def("compute_inverse_phase_matrix", &compute_inverse_phase_matrix,
        "Compute inverse phase matrix with high precision");

  // float32 implemetations
  m.def("compute_inverse_phase_matrix32", &compute_inverse_phase_matrix32,
        "Compute inverse phase matrix with float32 precision");
  m.def("compute_phase_matrix32", &compute_phase_matrix32,
        "Compute phase matrix with float32 precision");
}
