# Helper function to determine optimal chunk sizes

def get_optimal_chunk_sizes(num_pixels, num_vis, max_memory_gb=4, is_double=True):
    # Calculate memory per element
    bytes_per_element = 8 if is_double else 4

    # Calculate memory footprint for phase matrix of different sizes
    # We need 2 phase matrices (cos and sin) plus some overhead
    # Let's determine optimal chunk sizes that fit within max_memory_gb
    max_memory_bytes = max_memory_gb * 1024**3 * 0.9  # Use 80% of specified max memory

    # Strategy: Use larger visibility chunks and adjust pixel chunks
    # This makes better use of GPU parallelism
    vis_chunk_size = min(16384, num_vis)  # Use larger chunks for visibilities

    # Calculate max pixel chunk size that fits in memory
    # Phase matrix is vis_chunk_size × pixel_chunk_size
    # We need 2 such matrices (cos and sin) plus overhead
    max_elements = max_memory_bytes / (2 * bytes_per_element)
    pixel_chunk_size = int(max_elements / vis_chunk_size)
    pixel_chunk_size = min(pixel_chunk_size, num_pixels)
    pixel_chunk_size = max(1024, pixel_chunk_size)  # Ensure minimum size for efficiency
    return vis_chunk_size, pixel_chunk_size 
