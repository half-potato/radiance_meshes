import torch
from utils.topo_utils import build_tv_struct, tet_volumes
from utils.graphics_utils import l2_normalize_th

def calculate_gradient(value, model):
    value = value.reshape(-1)
    device = model.device
    verts = model.vertices.to(device)
    indices = model.indices.to(device)
    volumes = tet_volumes(verts[indices])
    num_tets = indices.shape[0]
    # Calculate the geometric center (centroid) of each tetrahedron
    tet_verts = verts[indices]  # Shape: (num_tets, 4, 3)
    centroids = torch.mean(tet_verts, dim=1) # Shape: (num_tets, 3)

    owners, face_areas = build_tv_struct(model.vertices, model.indices, device=device)
    t0_indices = owners[:, 0]
    t1_indices = owners[:, 1]

    # Part B: Calculate the terms for the Gradient Theorem
    # Get centroids and luminance for each tet in the pair
    c0 = centroids[t0_indices]
    c1 = centroids[t1_indices]
    lum0 = value[t0_indices]
    lum1 = value[t1_indices]

    # 1. Approximate Luminance at the face center (L_face)
    lum_face = (lum0 + lum1) / 2.0

    # 2. Calculate the outward-pointing unit normal vector for each face (n_face)
    # The vector from c0 to c1 serves as the normal pointing out of t0.
    centroid_vecs = c1 - c0
    face_normals = centroid_vecs / (torch.linalg.norm(centroid_vecs, dim=1, keepdim=True) + 1e-9)

    # 3. Calculate the "flux vector" for each face: L_face * A_face * n_face
    # This represents the contribution of each face to the surface integral.
    flux_vectors = lum_face.unsqueeze(1) * face_normals * face_areas.unsqueeze(1)

    # Part C: Sum the surface contributions and divide by volume
    num_tets = model.indices.shape[0]
    surface_integral_sum = torch.zeros(num_tets, 3, device=device)

    # Accumulate the flux vectors for each tetrahedron.
    # For t0, the flux vector points outward, so we add it.
    # For t1, the vector points inward, so we subtract it to get its outward contribution.
    surface_integral_sum.index_add_(0, t0_indices, flux_vectors)
    surface_integral_sum.index_add_(0, t1_indices, -flux_vectors)

    # Finally, divide by the volume of each tetrahedron to get the average gradient.
    grads = surface_integral_sum / (volumes.unsqueeze(1) + 1e-9)
    return grads
