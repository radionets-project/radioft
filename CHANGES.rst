Radioft 0.1.1 (2025-12-09)
==========================


API Changes
-----------


Bug Fixes
---------


Data Model Changes
------------------


New Features
------------

- Added coordinate grids for `lm` and `uvw_dense` with Earth curvature handling in `w`

  Added `FINUFFTAutograd` and `IFINUFFTAutograd` custom PyTorch autograd functions to enable differentiable NUFFT and inverse NUFFT, using the adjoint operation for backward passes. Exposed these via `apply_nufft_differentiable` and `apply_inufft_differentiable` functions.

  Introduced `cupy_to_torch` and `torch_to_cupy` utility functions for efficient conversion between CuPy arrays and PyTorch tensors using DLPack, ensuring all data stays on the GPU.

  Added the `_compute_visibility_weights_and_indices` method to count how many visibilities map to each pixel, and used these weights to normalize visibilities in the inverse NUFFT. This ensures correct averaging when multiple visibilities fall into the same bin. [`#23 <https://github.com/radionets-project/radioft/pull/23>`__]


Maintenance
-----------


Refactoring and Optimization
----------------------------

Radioft 0.1.1 (2025-12-09)
==========================


API Changes
-----------


Bug Fixes
---------


Data Model Changes
------------------


New Features
------------


Maintenance
-----------


Refactoring and Optimization
----------------------------

Radioft 0.1.0 (2025-12-09)
==========================


API Changes
-----------


Bug Fixes
---------

- Added `benchmark` flag in `radioft/dft/dft/HybridPyTorchCudaDFT` to fix seed setting and ensuring reproducible results only during benchmarking [`#10 <https://github.com/radionets-project/radioft/pull/10>`__]


Data Model Changes
------------------


New Features
------------

- Add dev environment file
- Add GitHub workflows [`#5 <https://github.com/radionets-project/radioft/pull/5>`__]

- Added wrapper for cufinufft to speed up non-uniform FFT calculations [`#12 <https://github.com/radionets-project/radioft/pull/12>`__]


Maintenance
-----------

- Moved radioft to src/radioft
- Updated pyproject.toml accordingly
- Added ruff linting and formatting rules
- Added .clang-format config [`#13 <https://github.com/radionets-project/radioft/pull/13>`__]

- Removed obsolete ``radioft.cuda`` module [`#20 <https://github.com/radionets-project/radioft/pull/20>`__]


Refactoring and Optimization
----------------------------

- Dependency and Configuration Updates

  - Updated `.pre-commit-config.yaml` to include the `ruff` linter and updated versions of `isort` and `flake8`. Removed `black` from the configuration.

- CUDA Kernels and Bindings Enhancements

  - Added new CUDA kernels (`compute_phase_kernel32` and `compute_inverse_phase_kernel32`) in `radioft/cuda/kernels/cuda_phase_kernel32.cu` to support float32 precision for phase matrix computations.
  - Refactored existing kernels in `radioft/cuda/kernels/cuda_phase_kernel.cu` to improve code formatting and readability, including consistent formatting for function signatures and kernel launches.
  - Added PyBind11 bindings in `radioft/cuda/cuda_bindings.cpp` to expose both float32 and float64 CUDA kernel functions to Python.

- PyTorch Integration and API Improvements

  - Updated `radioft/dft/dft.py` to add support for both float32 and float64 precision in the `HybridPyTorchCudaDFT` class. Introduced a `dtype` parameter in the constructor and `float64` flags in the `forward` and `inverse` methods to toggle precision.
  - Modified tensor initialization and typecasting in `forward` and `inverse` methods to dynamically set precision based on the `float64` flag. [`#6 <https://github.com/radionets-project/radioft/pull/6>`__]
