import os
import platform
import subprocess
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

# Check if CUDA is available
cuda_available = CUDA_HOME is not None

# Define CUDA extension
cuda_extensions = []
if cuda_available:
    compute_capability = os.environ.get("CUDA_COMPUTE_CAPABILITY")

    if compute_capability is None and platform.system() != "Windows":
        try:
            # See https://stackoverflow.com/a/71489432. We ommit the header
            # because we are only interested in the value
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]
            )
            # Strip value of "." so we get a value compatible with the
            # nvcc args below
            compute_capability = output.decode("utf-8").strip().replace(".", "")
        except (subprocess.SubprocessError, FileNotFoundError):
            # Default to a common compute capability if detection fails
            compute_capability = "80"  # Default to Ampere architecture

    if compute_capability is None:
        compute_capability = "80"  # Default to Ampere architecture

    # Find all CUDA source files in the radioft/cuda directory
    # cuda_sources = [str(p) for p in Path("radioft/cuda").glob("**/*.cu")]
    cuda_dir = Path("radioft/cuda")
    cuda_sources = list(cuda_dir.glob("**/*.cu"))
    cuda_sources.append(cuda_dir / "cuda_bindings.cpp")

    gencode = f"arch=compute_{compute_capability},code=sm_{compute_capability}"

    cuda_extensions = [
        CUDAExtension(
            name="radioft.cuda.kernels",  # Import as radioft.cuda.kernels
            sources=cuda_sources,  # Existing CUDA files in cada_dir
            include_dirs=[cuda_dir],  # Include directory for headers
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    f"-gencode={gencode}",
                    "-O3",
                ],
            },
        )
    ]


setup(
    name="radioft",
    packages=find_packages(),
    ext_modules=cuda_extensions,
    cmdclass={"build_ext": BuildExtension},
)
