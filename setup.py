import os
import platform
import subprocess
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

# Check if CUDA is available
cuda_available = CUDA_HOME is not None

# Define CUDA extension
cuda_extensions = []
if cuda_available:
    # Get CUDA compute capability if CUDA is available
    compute_capability = os.environ.get('CUDA_COMPUTE_CAPABILITY')

    # If not provided, try to determine automatically
    if compute_capability is None and platform.system() != 'Windows':
        try:
            # Run nvidia-smi to get compute capability
            output = subprocess.check_output(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'])
            compute_capability = output.decode('utf-8').strip().replace('.', '')
        except (subprocess.SubprocessError, FileNotFoundError):
            # Default to a common compute capability if detection fails
            compute_capability = '70'  # Default to Volta architecture

    # Use a default for Windows or if detection failed
    if compute_capability is None:
        compute_capability = '70'  # Default to Volta architecture

    # Find all CUDA source files in the radioft/cuda directory
    cuda_sources = []
    cuda_dir = os.path.join('radioft', 'cuda')
    for root, _, files in os.walk(cuda_dir):
        for file in files:
            if file.endswith(('.cu', '.cpp')):
                cuda_sources.append(os.path.join(root, file))

    # Define the CUDA extension using your existing files
    cuda_extensions = [
        CUDAExtension(
            name='radioft.cuda.kernels',  # Import as radioft.cuda.kernels
            sources=cuda_sources,         # Existing CUDA files in cada_dir
            include_dirs=[cuda_dir],      # Include directory for headers
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    f'-gencode=arch=compute_{compute_capability},code=sm_{compute_capability}',
                    '-O3',
                    '--use_fast_math'
                ]
            }
        )
    ]

# Setup package
setup(
    name="radioft",
    packages=find_packages(),
    ext_modules=cuda_extensions,
    cmdclass={
        'build_ext': BuildExtension
    },
)
