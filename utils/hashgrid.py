import torch
import torch.nn as nn
from icecream import ic

class HashEmbedderOptimized(nn.Module):
    def __init__(
        self,
        bounding_box,
        n_levels=16,
        n_features_per_level=2,
        log2_hashmap_size=19,
        base_resolution=16,
        finest_resolution=512,
        use_conditional_hashing=True  # Set this to True to enable the logic
    ):
        super(HashEmbedderOptimized, self).__init__()

        self.use_conditional_hashing = use_conditional_hashing
        box_min, box_max = bounding_box
        self.register_buffer("box_min", box_min)
        self.register_buffer("box_max", box_max)

        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.register_buffer("base_resolution", torch.tensor(base_resolution, dtype=torch.float32))
        self.register_buffer("finest_resolution", torch.tensor(finest_resolution, dtype=torch.float32))

        self.out_dim = n_levels * n_features_per_level

        self.b = torch.exp(
            (self.finest_resolution.log() - self.base_resolution.log()) / (n_levels - 1)
        )

        level_resolutions = []
        for i in range(n_levels):
            res_i = torch.floor(self.base_resolution * (self.b ** i))
            level_resolutions.append(res_i)
        self.register_buffer("level_resolutions", torch.stack(level_resolutions))

        # Dynamically size embedding tables based on the hashing strategy
        if not self.use_conditional_hashing:
            embedding_sizes = [2**log2_hashmap_size] * n_levels
        else:
            embedding_sizes = [
                min(2**log2_hashmap_size, int(res.item())**3)
                for res in self.level_resolutions
            ]

        self.embeddings = nn.ModuleList([
            nn.Embedding(size, n_features_per_level)
            for size in embedding_sizes
        ])

        for emb in self.embeddings:
            nn.init.uniform_(emb.weight, a=-0.0001, b=0.0001)

        prime_list = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]
        self.register_buffer("primes", torch.tensor(prime_list, dtype=torch.long))

        corners = torch.tensor([[i, j, k] for i in [0, 1] for j in [0, 1] for k in [0, 1]], dtype=torch.long)
        self.register_buffer("box_offsets", corners)

        self.hash_mask = (1 << log2_hashmap_size) - 1
        self.out_dim = n_levels * n_features_per_level
        self.n_output_dims = self.out_dim  # backwards compat

    def ingp_hash(self, coords):
        xor_result = torch.zeros_like(coords[..., 0])
        for i in range(coords.shape[-1]):
             xor_result ^= coords[..., i] * self.primes[i]
        return xor_result & self.hash_mask

    def get_voxel_vertices(self, x, level_idx):
        resolution = self.level_resolutions[level_idx]
        res_long = resolution.long()
        
        x_clamped = torch.clamp(x, min=self.box_min, max=self.box_max)
        scaled_coords = (x_clamped - self.box_min) * (resolution / (self.box_max - self.box_min))
        
        offset_coords = scaled_coords + 0.5

        bottom_left_idx = torch.floor(offset_coords).long()
        fracs = offset_coords - bottom_left_idx.float()

        voxel_indices = bottom_left_idx.unsqueeze(1) + self.box_offsets

        hashmap_size = 1 << self.log2_hashmap_size
        num_voxels = resolution ** 3

        if self.use_conditional_hashing and (num_voxels <= hashmap_size):
            # --- Collision-free path: Use modulo for wrap-around ---
            res_sq = res_long * res_long
            wrapped_indices = voxel_indices % res_long
            hashed_voxel_indices = (
                wrapped_indices[..., 0] + 
                wrapped_indices[..., 1] * res_long + 
                wrapped_indices[..., 2] * res_sq
            )
        else:
            # --- Standard iNGP hash: Use raw integer indices ---
            hashed_voxel_indices = self.ingp_hash(voxel_indices)
        
        return fracs, hashed_voxel_indices

    def trilinear_interp_direct(self, fracs, corner_embs):
        w = torch.stack([1.0 - fracs, fracs], dim=2)
        c0 = corner_embs[:, 0b000] * w[:,0,0][:,None] + corner_embs[:, 0b100] * w[:,0,1][:,None]
        c1 = corner_embs[:, 0b001] * w[:,0,0][:,None] + corner_embs[:, 0b101] * w[:,0,1][:,None]
        c2 = corner_embs[:, 0b010] * w[:,0,0][:,None] + corner_embs[:, 0b110] * w[:,0,1][:,None]
        c3 = corner_embs[:, 0b011] * w[:,0,0][:,None] + corner_embs[:, 0b111] * w[:,0,1][:,None]
        c0 = c0 * w[:,1,0][:,None] + c2 * w[:,1,1][:,None]
        c1 = c1 * w[:,1,0][:,None] + c3 * w[:,1,1][:,None]
        c = c0 * w[:,2,0][:,None] + c1 * w[:,2,1][:,None]
        return c

    def forward(self, x):
        x_embedded_all = []
        for i in range(self.n_levels):
            fracs, hashed_idxs = self.get_voxel_vertices(x, i)
            corner_embs = self.embeddings[i](hashed_idxs)
            x_embedded = self.trilinear_interp_direct(fracs, corner_embs)
            x_embedded_all.append(x_embedded)
        return torch.cat(x_embedded_all, dim=-1)

    def forward_in_chunks(self, x, chunk_size=548576):
        """
        Same as forward(), but processes 'x' in chunks to reduce memory usage.
        """
        outputs = []
        start = 0
        while start < x.shape[0]:
            end = min(start + chunk_size, x.shape[0])
            x_chunk = x[start:end]
            outputs.append(self.forward(x_chunk))
            start = end
        return torch.cat(outputs, dim=0)
