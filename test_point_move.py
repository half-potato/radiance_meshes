import torch
import numpy as np
import matplotlib.pyplot as plt
import mediapy as media

# Compute the circumcenter of a triangle with vertices A, B, C.
def compute_circumcenter(A, B, C):
    # A, B, C are 2D torch tensors.
    d = 2 * (A[0]*(B[1]-C[1]) + B[0]*(C[1]-A[1]) + C[0]*(A[1]-B[1]))
    Ux = ((A[0]**2 + A[1]**2)*(B[1]-C[1]) + (B[0]**2+B[1]**2)*(C[1]-A[1]) + (C[0]**2+C[1]**2)*(A[1]-B[1])) / d
    Uy = ((A[0]**2 + A[1]**2)*(C[0]-B[0]) + (B[0]**2+B[1]**2)*(A[0]-C[0]) + (C[0]**2+C[1]**2)*(B[0]-A[0])) / d
    return torch.stack([Ux, Uy])

# Given a point P and triangle vertices A, B, C, compute the barycentric coordinates of P.
def barycentric_coordinates(P, A, B, C):
    # Using the standard formula based on areas.
    v0 = B - A
    v1 = C - A
    v2 = P - A
    d00 = torch.dot(v0, v0)
    d01 = torch.dot(v0, v1)
    d11 = torch.dot(v1, v1)
    d20 = torch.dot(v2, v0)
    d21 = torch.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    # v and w are the weights for vertices B and C
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w
    return torch.tensor([u, v, w])

# Clip barycentric coordinates to [0,1] and renormalize so they sum to 1.
def clip_barycentrics(bary):
    clipped = torch.clamp(bary, 0, 1)
    # clipped = bary
    s = clipped.sum()
    if s > 0:
        clipped = clipped / s
    return clipped

# Create a list to hold video frames.
frames = []
num_frames = 120

# Animate the triangle by varying its vertices over time.
for i in range(num_frames):
    t = 2 * np.pi * i / num_frames
    t_tensor = torch.tensor(t)
    # Define triangle vertices.
    # Here the vertices move over time in a non-symmetric way so that sometimes the triangle is obtuse.
    A = torch.tensor([0.5 + 0.1 * torch.sin(t_tensor), 0.5 + 0.3 * torch.cos(t_tensor)])
    B = torch.tensor([0.5 + 0.1 * torch.sin(t_tensor + 0.123*np.pi/3), 0.5 + 0.3 * torch.cos(t_tensor + 8*np.pi/3)])
    # Adding an extra modulation to C to induce asymmetry.
    C = torch.tensor([
        0.5 + 0.3 * torch.sin(t_tensor + 4*np.pi/3 + 0.3 * torch.sin(2*t_tensor)),
        0.5 + 0.1 * torch.cos(t_tensor + 4*np.pi/3 + 0.3 * torch.cos(2*t_tensor))
    ])

    # Compute the circumcenter.
    circ = compute_circumcenter(A, B, C)
    # Compute its barycentric coordinates.
    bary = barycentric_coordinates(circ, A, B, C)
    # Clip these barycentrics and renormalize.
    clipped_bary = clip_barycentrics(bary)
    # Reconstruct the clipped circumcenter from the clipped barycentrics.
    clipped_circ = A * clipped_bary[0] + B * clipped_bary[1] + C * clipped_bary[2]

    # Plot the triangle and the two centers.
    fig, ax = plt.subplots(figsize=(4, 4))
    # Plot the triangle (cycle back to A to close the triangle).
    triangle = torch.stack([A, B, C, A]).numpy()
    ax.plot(triangle[:, 0], triangle[:, 1], 'k-', lw=2)
    ax.plot(triangle[:-1, 0], triangle[:-1, 1], 'ko', markersize=5)
    # Plot the circumcenter in red.
    ax.plot(circ[0].item(), circ[1].item(), 'ro', label='Circumcenter')
    # Plot the clipped circumcenter in blue.
    ax.plot(clipped_circ[0].item(), clipped_circ[1].item(), 'bo', label='Clipped Circumcenter')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(f'Frame {i}')

    # Convert the plot to an image array.
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frames.append(img)
    plt.close(fig)

# Write the video using mediapy.
media.write_video('triangle_animation.mp4', np.array(frames), fps=30)
print("Video saved as triangle_animation.mp4")
