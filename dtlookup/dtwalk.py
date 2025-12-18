import numpy as np
import time
from scipy.spatial import Delaunay # Only used for the demo

class TetWalkLookup:
    """
    Combines the fast pre-inverted matrix lookup with a
    grid-based starting point cache to minimize walk length.
    
    This has the SLOWEST __init__ (adjacency + all inverses + grid build)
    but should have the FASTEST possible lookup.
    """
    
    def __init__(self, indices: np.ndarray, vertices: np.ndarray, grid_res: int = 20):
        """
        Initializes by building adjacency, pre-inverting all matrices,
        and building a grid of starting tetrahedra.
        
        Args:
            indices (np.ndarray): (N_tets, 4) int array of tet vertex indices.
            vertices (np.ndarray): (N_verts, 3) float array of vertex positions.
            grid_res (int): The resolution of the acceleration grid (e.g., 20x20x20).
                            Higher is slower to init, but faster to lookup.
        """
        print(f"Building GridAcceleratedTetWalkLookup (grid_res={grid_res})...")
        self.indices = indices.astype(np.int64)
        self.vertices = vertices.astype(np.float64)
        self.n_tets = self.indices.shape[0]
        self.grid_res = grid_res

        # --- 1. Build Adjacency Graph (from TetWalkLookup) ---
        print("  1. Building adjacency graph...")
        self.neighbors = -np.ones((self.n_tets, 4), dtype=np.int64)
        face_to_tet_map = {}
        face_indices = [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)]
        for i in range(self.n_tets):
            tet_verts = self.indices[i]
            for j in range(4):
                face_j_verts = tet_verts[list(face_indices[j])]
                face_key = tuple(sorted(face_j_verts))
                if face_key in face_to_tet_map:
                    other_tet_idx, other_face_idx = face_to_tet_map[face_key]
                    self.neighbors[i, j] = other_tet_idx
                    self.neighbors[other_tet_idx, other_face_idx] = i
                    del face_to_tet_map[face_key]
                else:
                    face_to_tet_map[face_key] = (i, j)

        # --- 2. Pre-compute all T-inverses (from TetWalkLookup) ---
        print("  2. Pre-inverting all T matrices...")
        tets_verts = self.vertices[self.indices]
        T = np.ones((self.n_tets, 4, 4), dtype=np.float64)
        T[:, :3, :] = np.transpose(tets_verts, (0, 2, 1))
        try:
            self.T_inv = np.linalg.inv(T)
        except np.linalg.LinAlgError: # Handle degenerate tets
            self.T_inv = np.zeros_like(T)
            for i in range(self.n_tets):
                try:
                    self.T_inv[i] = np.linalg.inv(T[i])
                except np.linalg.LinAlgError:
                    pass
        
        # --- 3. Build the Starting Point Grid (NEW) ---
        print("  3. Building acceleration grid...")
        self.start_grid = -np.ones((grid_res, grid_res, grid_res), dtype=np.int64)
        
        # Calculate world bounds with padding
        self.world_min = np.min(self.vertices, axis=0)
        self.world_max = np.max(self.vertices, axis=0)
        padding = (self.world_max - self.world_min) * 0.01
        self.world_min -= padding
        self.world_max += padding
        
        # Ensure world_max is slightly larger to avoid index-out-of-bounds
        self.cell_size = (self.world_max - self.world_min) / grid_res
        
        # We need a temporary "lookup" function to populate the grid
        # This is the same as the final lookup, but with a fixed start
        def _temp_lookup(point, start_tet=0):
            current_tet_idx = start_tet
            for _ in range(self.n_tets + 1): 
                bary = self._get_barycentric(point, current_tet_idx)
                if np.all(bary >= -1e-9):
                    return current_tet_idx # Found it
                face_to_cross = np.argmin(bary)
                next_tet_idx = self.neighbors[current_tet_idx, face_to_cross]
                if next_tet_idx == -1:
                    return -1 # Outside mesh
                current_tet_idx = next_tet_idx
            return -1 # Walk failed

        # Iterate over all grid cells and find a tet for each
        # We start the search from the previously found tet for speed
        print("Building grid")
        last_found_tet = 0
        for i in range(grid_res):
            for j in range(grid_res):
                for k in range(grid_res):
                    # Get cell center
                    cell_min = self.world_min + np.array([i, j, k]) * self.cell_size
                    cell_center = cell_min + self.cell_size * 0.5
                    
                    # Find the tet for this cell center
                    start_tet = _temp_lookup(cell_center, last_found_tet)
                    self.start_grid[i, j, k] = start_tet
                    if start_tet != -1:
                        last_found_tet = start_tet
        
        # This will be used as a fallback
        self.global_start_tet = last_found_tet
        if self.global_start_tet == -1:
             self.global_start_tet = 0 # Failsafe
        
        print("...Build complete.")

    def _get_barycentric(self, point: np.ndarray, tet_idx: int) -> np.ndarray:
        """Calculates barycentric coordinates using pre-computed inverse."""
        # This check is needed because the tet_idx might be -1
        if tet_idx == -1:
            return np.array([-1.0, -1.0, -1.0, -1.0])
        p_h = np.append(point, 1.0)
        bary = self.T_inv[tet_idx] @ p_h
        return bary

    def lookup(self, point: np.ndarray) -> int:
        """
        Finds the tetrahedron containing the point using the
        acceleration grid to find an optimal starting point.
        """
        
        # --- 1. Find optimal start_tet_idx from grid ---
        point = point.reshape(3)
        
        # Calculate grid coordinates
        coords = (point - self.world_min) / self.cell_size
        
        # Floor and clamp to grid bounds
        grid_idx = np.floor(coords).astype(np.int64)
        grid_idx = np.clip(grid_idx, 0, self.grid_res - 1)
        
        # Get the pre-computed start tet
        current_tet_idx = self.start_grid[tuple(grid_idx)].item()
        
        # If the grid cell was outside the mesh, fall back
        if current_tet_idx == -1:
            current_tet_idx = self.global_start_tet # Use a known-good start tet
            
        # --- 2. Perform the walk (should be very short) ---
        
        # The walk loop. We must check the first tet, as the point might
        # be in a different cell than its optimal start tet (if it's near
        # a cell boundary).
        for _ in range(self.n_tets + 1): 
            
            bary = self._get_barycentric(point, current_tet_idx)
            
            if np.all(bary >= -1e-9):
                return current_tet_idx
            
            face_to_cross = np.argmin(bary)
            next_tet_idx = self.neighbors[current_tet_idx, face_to_cross]
            
            if next_tet_idx == -1:
                return -1
            
            current_tet_idx = next_tet_idx

        return -1 # Walk failed

# --- Example Usage & Performance Comparison ---

if __name__ == "__main__":
    
    print("Generating demo data...")
    num_verts = 5000 
    
    points_np = np.random.rand(num_verts, 3)
    delaunay = Delaunay(points_np)
    
    vertices = delaunay.points.astype(np.float64)
    indices = delaunay.simplices.astype(np.int64)
    
    print(f"Generated {vertices.shape[0]} vertices and {indices.shape[0]} tets.")

    # --- 1. Time the NEW (Grid Accelerated) class ---
    print("\n--- Testing GridAcceleratedTetWalkLookup (Slowest Init, Fastest Lookup) ---")
    
    start_init = time.perf_counter()
    # A 20x20x20 grid (8000 cells) is a good start
    lookup_grid = GridAcceleratedTetWalkLookup(indices, vertices, grid_res=20)
    end_init = time.perf_counter()
    print(f"\n__init__ time:      {end_init - start_init:.6f} s (Very Slow!)")

    # Time a lookup
    query_point_1 = np.array([0.5, 0.5, 0.5])
    start_lookup = time.perf_counter()
    result_1 = lookup_grid.lookup(query_point_1)
    end_lookup = time.perf_counter()
    
    print(f"\nLookup 1 (Random): {(end_lookup - start_lookup) * 1e6:.2f} µs")
    print(f"Found tet: {result_1}")

    # Time another lookup far away
    query_point_2 = np.array([0.9, 0.9, 0.9])
    start_lookup = time.perf_counter()
    result_2 = lookup_grid.lookup(query_point_2)
    end_lookup = time.perf_counter()
    
    print(f"\nLookup 2 (Random): {(end_lookup - start_lookup) * 1e6:.2f} µs (Should be similar speed)")
    print(f"Found tet: {result_2}")

    # --- 2. Time the original (Slow Init, Fast Walk) class for comparison ---
    print("\n--- Testing TetWalkLookup (Slow Init, but no grid) ---")
    
    start_init_slow = time.perf_counter()
    lookup_slow = TetWalkLookup(indices, vertices) # From previous code block
    end_init_slow = time.perf_counter()
    print(f"__init__ time:      {end_init_slow - start_init_slow:.6f} s (Slow, but faster than grid)")
    
    # Time lookup 1 (starts at 0, has to walk far)
    start_lookup_slow_1 = time.perf_counter()
    result_slow_1 = lookup_slow.lookup(query_point_1, start_tet_idx=0)
    end_lookup_slow_1 = time.perf_counter()
    print(f"\nLookup 1 (Walk from 0): {(end_lookup_slow_1 - start_lookup_slow_1) * 1e6:.2f} µs")

    # Time lookup 2 (starts at 0, has to walk far)
    start_lookup_slow_2 = time.perf_counter()
    result_slow_2 = lookup_slow.lookup(query_point_2, start_tet_idx=0)
    end_lookup_slow_2 = time.perf_counter()
    print(f"\nLookup 2 (Walk from 0): {(end_lookup_slow_2 - start_lookup_slow_2) * 1e6:.2f} µs (Probably slow)")
    
    # Time lookup 3 (starts from last point, should be fast)
    start_lookup_slow_3 = time.perf_counter()
    result_slow_3 = lookup_slow.lookup(np.array([0.9, 0.9, 0.91])) # Point near last
    end_lookup_slow_3 = time.perf_counter()
    print(f"\nLookup 3 (Cached Walk): {(end_lookup_slow_3 - start_lookup_slow_3) * 1e6:.2f} µs (Fast, but only if close)")
    
    
    # --- Verification ---
    scipy_res_1 = delaunay.find_simplex(query_point_1)
    scipy_res_2 = delaunay.find_simplex(query_point_2)
    print(f"\n--- Verification ---")
    print(f"Grid 1:   {result_1} (Scipy: {scipy_res_1}) -> Match: {result_1 == scipy_res_1}")
    print(f"Grid 2:   {result_2} (Scipy: {scipy_res_2}) -> Match: {result_2 == scipy_res_2}")