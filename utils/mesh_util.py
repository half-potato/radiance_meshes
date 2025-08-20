import numpy as np
from collections import deque, defaultdict
# from skimage.draw import polygon, polygon_perimeter
from PIL import Image
from pathlib import Path
# import xatlas
# import trimesh

def tet_to_vert_color(verts, tets, tet_v_rgb):
    n_verts = len(verts)
    n_ch = tet_v_rgb.shape[-1]

    v_rgb = np.zeros((n_verts, n_ch), dtype=np.float32)
    flat_t_idx = tets.flatten()
    flat_t_rgb = tet_v_rgb.reshape(-1, n_ch)

    np.add.at(v_rgb, flat_t_idx, flat_t_rgb)
    
    v_counts = np.bincount(flat_t_idx, minlength=n_verts)
    
    non_zero = v_counts > 0
    v_rgb[non_zero] /= v_counts[non_zero, np.newaxis]

    return v_rgb

def extract_meshes(rgb, verts, tets):
    # 1. Pre-calculate all vertex colors correctly
    v_rgb = tet_to_vert_color(verts, tets, rgb)

    # 2. Build tetrahedron adjacency graph
    face_map = defaultdict(list)
    all_faces = np.stack([
        tets[:, [1, 3, 2]], tets[:, [0, 2, 3]],
        tets[:, [0, 3, 1]], tets[:, [0, 1, 2]],
    ], axis=1)

    for ti, tet_faces in enumerate(all_faces):
        for face in tet_faces:
            key = tuple(sorted(face))
            face_map[key].append(ti)
    
    tet_adj = defaultdict(list)
    for key, t_indices in face_map.items():
        if len(t_indices) == 2:
            t0, t1 = t_indices
            tet_adj[t0].append(t1)
            tet_adj[t1].append(t0)

    # 3. Flood-fill across tetrahedra to find connected volumes
    components, seen = [], set()
    for i in range(len(tets)):
        if i in seen:
            continue
        comp = []
        q = deque([i])
        while q:
            ti = q.popleft()
            if ti in seen:
                continue
            seen.add(ti)
            comp.append(ti)
            q.extend(tet_adj[ti])
        components.append(np.array(comp, dtype=np.int64))

    # 4. Extract boundary mesh from each volume
    meshes = []
    for comp_t_indices in components:
        comp_tets = tets[comp_t_indices]
        
        comp_faces = np.stack([
            comp_tets[:, [1, 3, 2]], comp_tets[:, [0, 2, 3]],
            comp_tets[:, [0, 3, 1]], comp_tets[:, [0, 1, 2]],
        ], axis=1).reshape(-1, 3)

        faces_key = np.sort(comp_faces, axis=1)
        keys, inv, counts = np.unique(
            faces_key, axis=0, return_inverse=True, return_counts=True
        )
        
        b_mask = counts[inv] == 1
        b_faces = comp_faces[b_mask]

        if len(b_faces) == 0:
            continue
            
        unique_vs, new_idx = np.unique(b_faces, return_inverse=True)
        
        meshes.append(dict(
            vertex=dict(
                x=verts[unique_vs, 0].astype(np.float32),
                y=verts[unique_vs, 1].astype(np.float32),
                z=verts[unique_vs, 2].astype(np.float32),
                r=v_rgb[unique_vs, 0].astype(np.float32),
                g=v_rgb[unique_vs, 1].astype(np.float32),
                b=v_rgb[unique_vs, 2].astype(np.float32),
            ),
            face=dict(
                vertex_indices=new_idx.reshape(-1, 3).astype(np.int32)
            )
        ))
        
    return meshes

def extract_meshes_per_face_color(rgb, verts, tets):
    n_ch = rgb.shape[-1]
    color_names = ['r', 'g', 'b', 'a'][:n_ch]

    # Step 1 & 2: Find connected components (This part is correct)
    face_definitions = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]], dtype=np.int32)
    all_faces = tets[:, face_definitions].reshape(-1, 3)
    
    face_map = defaultdict(list)
    for i in range(len(all_faces)):
        ti, _ = divmod(i, 4)
        key = tuple(sorted(all_faces[i]))
        face_map[key].append(ti)

    tet_adj = defaultdict(list)
    for key, t_indices in face_map.items():
        if len(t_indices) == 2:
            t0, t1 = t_indices
            tet_adj[t0].append(t1)
            tet_adj[t1].append(t0)
            
    components, seen = [], set()
    for i in range(len(tets)):
        if i in seen: continue
        comp = []; q = deque([i])
        while q:
            ti = q.popleft()
            if ti in seen: continue
            seen.add(ti); comp.append(ti); q.extend(tet_adj[ti])
        components.append(np.array(comp, dtype=np.int64))

    # Step 3: Extract boundary meshes
    meshes = []
    for comp_t_indices in components:
        comp_tets = tets[comp_t_indices]
        comp_rgb = rgb[comp_t_indices]
        
        comp_all_faces = comp_tets[:, face_definitions].reshape(-1, 3)
        face_tet_idx = np.arange(len(comp_tets)).repeat(4)
        
        faces_key = np.sort(comp_all_faces, axis=1)
        _, inv, counts = np.unique(faces_key, axis=0, return_inverse=True, return_counts=True)
        
        b_mask = counts[inv] == 1
        b_faces = comp_all_faces[b_mask]
        b_face_tet_idx = face_tet_idx[b_mask]

        if len(b_faces) == 0:
            continue
            
        # --- NEW, EXPLICIT COLOR LOOKUP ---
        n_new_verts = len(b_faces) * 3
        new_verts_pos = np.zeros((n_new_verts, 3), dtype=np.float32)
        new_verts_rgb = np.zeros((n_new_verts, n_ch), dtype=np.float32)
        new_f_idx = np.arange(n_new_verts).reshape(-1, 3)

        v_counter = 0
        for i in range(len(b_faces)):
            face_global_indices = b_faces[i]
            tet_local_idx = b_face_tet_idx[i]

            source_tet_v_indices = comp_tets[tet_local_idx]
            source_tet_colors = comp_rgb[tet_local_idx]

            for v_global_idx in face_global_indices:
                # Find the local index (0-3) of the vertex in the tetrahedron
                local_v_idx = np.where(source_tet_v_indices == v_global_idx)[0][0]

                # Assign the correct position and color
                new_verts_pos[v_counter] = verts[v_global_idx]
                new_verts_rgb[v_counter] = source_tet_colors[local_v_idx]
                v_counter += 1
        # --- END NEW LOGIC ---

        v_dict = dict(
            x=new_verts_pos[:, 0].astype(np.float32),
            y=new_verts_pos[:, 1].astype(np.float32),
            z=new_verts_pos[:, 2].astype(np.float32),
        )
        for i, name in enumerate(color_names):
            v_dict[name] = new_verts_rgb[:, i].astype(np.float32)

        meshes.append(dict(
            vertex=v_dict,
            face=dict(
                vertex_indices=new_f_idx.astype(np.int32)
            )
        ))
            
    return meshes
