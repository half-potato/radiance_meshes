import torch
import slangtorch
import numpy as np
import time
import os
from scipy.spatial import Delaunay

def morton_encode(coords):
    """Encodes 3D integer coordinates into 64-bit Morton codes."""
    # Based on the C++ implementation from Wikipedia
    def part1by2(n):
        n = n.astype(np.int64) # Cast to 64-bit to prevent overflow during bit shifts
        n &= 0x00000000001fffff
        n = (n | (n << 32)) & 0x001f00000000ffff
        n = (n | (n << 16)) & 0x001f0000ff0000ff
        n = (n | (n << 8))  & 0x100f00f00f00f00f
        n = (n | (n << 4))  & 0x10c30c30c30c30c3
        n = (n | (n << 2))  & 0x1249249249249249
        return n

    x = part1by2(coords[:, 0])
    y = part1by2(coords[:, 1])
    z = part1by2(coords[:, 2])
    return x | (y << 1) | (z << 2)

# --- 0. Load Slang Modules (Loaded once globally) ---
shaders_path = os.path.join(os.path.dirname(__file__), "slang")
tet_splatter_module = slangtorch.loadModule(os.path.join(shaders_path, "tet_splatter.slang"))
key_generator_module = slangtorch.loadModule(os.path.join(shaders_path, "key_generator.slang"))
lookup_module = slangtorch.loadModule(os.path.join(shaders_path, "lookup_kernel.slang"))

class TetrahedraLookup:
    """
    Pre-computes an acceleration structure for a fixed set of tetrahedra
    to allow for repeated, fast point containment lookups.
    """
    def __init__(self, indices: torch.Tensor, vertices: torch.Tensor, grid_dim_size: int = 128):
        """
        Initializes the lookup structure by building the grid.

        Args:
            indices (torch.Tensor): (N_tets, 4) tensor of tetrahedron vertex indices.
            vertices (torch.Tensor): (N_verts, 3) tensor of vertex positions.
            grid_dim_size (int): The resolution of the acceleration grid (e.g., 128).
        """
        self.indices = indices.int() # Ensure correct type
        self.vertices = vertices.float() # Ensure correct type
        self.device = vertices.device
        self.n_tets = indices.shape[0]

        # --- 1. Determine Grid Parameters ---
        self.grid_dim = torch.tensor([grid_dim_size] * 3, device=self.device, dtype=torch.int32)
        
        world_min, _ = torch.min(vertices, dim=0)
        world_max, _ = torch.max(vertices, dim=0)
        padding = (world_max - world_min) * 0.01
        self.world_min = world_min - padding
        self.world_max = world_max + padding

        self.cell_size = (self.world_max - self.world_min) / (self.grid_dim.float())
        self.grid_size = self.grid_dim[0] * self.grid_dim[1] * self.grid_dim[2]
        
        # --- 2. Build the Grid Acceleration Structure ---
        tiles_touched = torch.zeros(self.n_tets, device=self.device, dtype=torch.int32)
        cell_rect = torch.zeros((self.n_tets, 6), device=self.device, dtype=torch.int32)

        tet_splatter_module.splat_tetrahedra(
            indices=self.indices,
            vertices=self.vertices,
            world_min=self.world_min.tolist(),
            cell_size=self.cell_size.tolist(),
            grid_dim=self.grid_dim.tolist(),
            out_tiles_touched=tiles_touched,
            out_cell_rect=cell_rect
        ).launchRaw(blockSize=(256, 1, 1), gridSize=((self.n_tets + 255) // 256, 1, 1))

        index_buffer_offset = torch.cumsum(tiles_touched, dim=0, dtype=torch.int32)
        self.total_keys = index_buffer_offset[-1].item() if self.n_tets > 0 else 0

        if self.total_keys > 0:
            unsorted_keys = torch.empty(self.total_keys, device=self.device, dtype=torch.int32)
            unsorted_tet_idx = torch.empty(self.total_keys, device=self.device, dtype=torch.int32)

            key_generator_module.generate_keys(
                cell_rect=cell_rect, index_buffer_offset=index_buffer_offset, grid_dim=self.grid_dim.tolist(),
                out_unsorted_keys=unsorted_keys, out_unsorted_tet_idx=unsorted_tet_idx
            ).launchRaw(blockSize=(256, 1, 1), gridSize=((self.n_tets + 255) // 256, 1, 1))

            self.sorted_keys, sort_indices = torch.sort(unsorted_keys)
            self.sorted_tet_idx = unsorted_tet_idx[sort_indices]

            self.tile_ranges = torch.zeros((self.grid_size, 2), device=self.device, dtype=torch.int32)
            key_generator_module.compute_tile_ranges(
                sorted_keys=self.sorted_keys, out_tile_ranges=self.tile_ranges
            ).launchRaw(blockSize=(256, 1, 1), gridSize=((self.total_keys + 255) // 256, 1, 1))
        else: # Handle case with no tetrahedra
            self.sorted_keys = None
            self.sorted_tet_idx = None
            self.tile_ranges = None


    def lookup(self, points: torch.Tensor) -> torch.Tensor:
        """
        Performs a massively parallel lookup using the pre-built structure.

        Args:
            points (torch.Tensor): (N_queries, 3) tensor of query points.

        Returns:
            torch.Tensor: (N_queries,) tensor containing the index of the containing
                          tetrahedron for each point, or -1 if not found.
        """
        n_queries = points.shape[0]
        if n_queries == 0:
            return torch.empty(0, dtype=torch.int32, device=self.device)

        # --- 3. Sort Query Points by Morton Code ---
        with torch.no_grad():
            int_coords = ((points - self.world_min) / self.cell_size).clamp(min=0).to(torch.int32)
            # Handle potential OOB on the max side
            int_coords = torch.min(int_coords, self.grid_dim - 1)
            
            morton_codes = torch.from_numpy(
                morton_encode(int_coords.cpu().numpy())
            ).to(self.device)
            
            morton_sort_indices = torch.argsort(morton_codes)
            sorted_query_points = points[morton_sort_indices]

        # --- 4. Run the Lookup Kernel ---
        output_tet_ids = torch.full((n_queries,), -1, dtype=torch.int32, device=self.device)
        if self.total_keys > 0 and n_queries > 0:
            lookup_module.lookup_points(
                query_points=sorted_query_points, 
                indices=self.indices, 
                vertices=self.vertices,
                sorted_keys=self.sorted_keys, 
                sorted_tet_idx=self.sorted_tet_idx,
                tile_ranges=self.tile_ranges, 
                world_min=self.world_min.tolist(), 
                cell_size=self.cell_size.tolist(),
                grid_dim=self.grid_dim.tolist(), 
                out_tet_ids=output_tet_ids
            ).launchRaw(blockSize=(256, 1, 1), gridSize=((n_queries + 255) // 256, 1, 1))

        # --- 5. Un-sort Results and Return ---
        inverse_sort_indices = torch.empty_like(morton_sort_indices)
        inverse_sort_indices[morton_sort_indices] = torch.arange(n_queries, device=self.device)
        final_results = output_tet_ids[inverse_sort_indices]
        
        return final_results

def lookup_inds(indices, vertices, points):
    """
    Performs a massively parallel lookup to find which tetrahedron contains each query point.

    Args:
        indices (torch.Tensor): (N_tets, 4) tensor of tetrahedron vertex indices.
        vertices (torch.Tensor): (N_verts, 3) tensor of vertex positions.
        points (torch.Tensor): (N_queries, 3) tensor of query points.

    Returns:
        torch.Tensor: (N_queries,) tensor containing the index of the containing tetrahedron
                      for each point, or -1 if not found.
    """
    device = vertices.device
    n_tets = indices.shape[0]
    n_queries = points.shape[0]

    # --- 1. Determine Grid Parameters from Input Data ---
    grid_dim = torch.tensor([128, 128, 128], device=device, dtype=torch.int32)
    
    # Calculate world bounds from vertices with a small padding
    world_min, _ = torch.min(vertices, dim=0)
    world_max, _ = torch.max(vertices, dim=0)
    padding = (world_max - world_min) * 0.01
    world_min -= padding
    world_max += padding

    cell_size = (world_max - world_min) / (grid_dim.float())
    grid_size = grid_dim[0] * grid_dim[1] * grid_dim[2]

    # --- 2. Build the Grid Acceleration Structure ---
    tiles_touched = torch.zeros(n_tets, device=device, dtype=torch.int32)
    cell_rect = torch.zeros((n_tets, 6), device=device, dtype=torch.int32)

    tet_splatter_module.splat_tetrahedra(
        indices=indices.int(),
        vertices=vertices.float(),
        world_min=world_min.tolist(),
        cell_size=cell_size.tolist(),
        grid_dim=grid_dim.tolist(),
        out_tiles_touched=tiles_touched,
        out_cell_rect=cell_rect
    ).launchRaw(blockSize=(256, 1, 1), gridSize=((n_tets + 255) // 256, 1, 1))

    index_buffer_offset = torch.cumsum(tiles_touched, dim=0, dtype=torch.int32)
    total_keys = index_buffer_offset[-1].item() if n_tets > 0 else 0

    if total_keys > 0:
        unsorted_keys = torch.empty(total_keys, device=device, dtype=torch.int32)
        unsorted_tet_idx = torch.empty(total_keys, device=device, dtype=torch.int32)

        key_generator_module.generate_keys(
            cell_rect=cell_rect, index_buffer_offset=index_buffer_offset, grid_dim=grid_dim.tolist(),
            out_unsorted_keys=unsorted_keys, out_unsorted_tet_idx=unsorted_tet_idx
        ).launchRaw(blockSize=(256, 1, 1), gridSize=((n_tets + 255) // 256, 1, 1))

        sorted_keys, sort_indices = torch.sort(unsorted_keys)
        sorted_tet_idx = unsorted_tet_idx[sort_indices]

        tile_ranges = torch.zeros((grid_size, 2), device=device, dtype=torch.int32)
        key_generator_module.compute_tile_ranges(
            sorted_keys=sorted_keys, out_tile_ranges=tile_ranges
        ).launchRaw(blockSize=(256, 1, 1), gridSize=((total_keys + 255) // 256, 1, 1))
    else: # Handle case with no tetrahedra
        sorted_keys, sorted_tet_idx, tile_ranges = None, None, None

    # --- 3. Sort Query Points by Morton Code ---
    with torch.no_grad():
        int_coords = ((points - world_min) / cell_size).clamp(min=0).to(torch.int32).cpu().numpy()
        morton_codes = torch.from_numpy(morton_encode(int_coords)).to(device)
        morton_sort_indices = torch.argsort(morton_codes)
        sorted_query_points = points[morton_sort_indices]

    # --- 4. Run the Lookup Kernel ---
    output_tet_ids = torch.full((n_queries,), -1, dtype=torch.int32, device=device)
    if total_keys > 0 and n_queries > 0:
        lookup_module.lookup_points(
            query_points=sorted_query_points, indices=indices, vertices=vertices,
            sorted_keys=sorted_keys, sorted_tet_idx=sorted_tet_idx,
            tile_ranges=tile_ranges, world_min=world_min.tolist(), cell_size=cell_size.tolist(),
            grid_dim=grid_dim.tolist(), out_tet_ids=output_tet_ids
        ).launchRaw(blockSize=(256, 1, 1), gridSize=((n_queries + 255) // 256, 1, 1))

    # --- 5. Un-sort Results and Return ---
    inverse_sort_indices = torch.empty_like(morton_sort_indices)
    inverse_sort_indices[morton_sort_indices] = torch.arange(n_queries, device=device)
    final_results = output_tet_ids[inverse_sort_indices]
    
    return final_results

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 1. Create a Triangulation with SciPy ---
    print("--- Creating SciPy Delaunay Triangulation ---")
    np.random.seed(42)
    # Generate points for the triangulation (the vertices)
    initial_points = np.random.rand(30, 3) 
    tri = Delaunay(initial_points)

    # Get triangulation data
    d_vertices_np = tri.points
    d_indices_np = tri.simplices

    print(f"Generated a triangulation with {d_vertices_np.shape[0]} vertices and {d_indices_np.shape[0]} tetrahedra.")

    # --- 2. Create Query Points ---
    # Create some points guaranteed to be inside by averaging vertices
    query_points_inside = d_vertices_np[d_indices_np[:5]].mean(axis=1)
    # Create some random points that may be inside or outside
    query_points_random = np.random.rand(500, 3)
    query_points_np = np.vstack([query_points_inside, query_points_random])

    # --- 3. Perform CPU Lookup with SciPy ---
    print("\n--- Performing Lookup with SciPy (CPU) ---")
    cpu_results = tri.find_simplex(query_points_np)

    # --- 4. Perform GPU Lookup with our Function ---
    print("--- Performing Lookup with SlangTorch (GPU) ---")
    # Convert numpy arrays to torch tensors on the GPU
    vertices_gpu = torch.from_numpy(d_vertices_np).float().to(device)
    indices_gpu = torch.from_numpy(d_indices_np).int().to(device)
    points_gpu = torch.from_numpy(query_points_np).float().to(device)

    # Call our lookup function
    gpu_results_gpu = lookup_inds(indices_gpu, vertices_gpu, points_gpu)
    gpu_results = gpu_results_gpu.cpu().numpy() # Move results back to CPU for comparison

    # --- 5. Compare Results ---
    print("\n--- Comparison of Results ---")
    print(f"{'Point #':<10}{'Query Point (X, Y, Z)':<30}{'SciPy (CPU)':<15}{'Our GPU Code':<15}{'Match?':<10}")
    print("-" * 80)

    for i in range(len(query_points_np)):
        p_str = f"({query_points_np[i,0]:.2f}, {query_points_np[i,1]:.2f}, {query_points_np[i,2]:.2f})"
        match_str = "✅" if cpu_results[i] == gpu_results[i] else "❌"
        print(f"{i:<10}{p_str:<30}{cpu_results[i]:<15}{gpu_results[i]:<15}{match_str:<10}")
        
    if np.array_equal(cpu_results, gpu_results):
        print("\n✅ All results match perfectly!")
    else:
        print("\n❌ Mismatch found between CPU and GPU results.")

