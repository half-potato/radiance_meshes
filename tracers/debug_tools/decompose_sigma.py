import pytorch3d.transforms
import torch
from icecream import ic
from scipy.spatial.transform import Rotation

import contractions
import util


def quat2mat(q):
    # q2 = torch.cat([q[1:], q[:1]])
    # print(q2, q)
    r = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    return torch.tensor(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - r * z),
            2.0 * (x * z + r * y),
            2.0 * (x * y + r * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - r * x),
            2.0 * (x * z - r * y),
            2.0 * (y * z + r * x),
            1.0 - 2.0 * (x * x + y * y),
        ]
    ).reshape(3, 3)


quat = torch.rand(4)
quat /= torch.linalg.norm(quat)
scale = torch.rand(3)
R = quat2mat(quat)
S = torch.diag(scale)
M = R.T @ S @ S @ R

eig = torch.linalg.eig(M)
ic(eig.eigenvalues, eig.eigenvectors)
scales2 = eig.eigenvalues.real.sqrt()
R2 = eig.eigenvectors.real.T
R2 = R2 * torch.det(R2)
S2 = torch.diag(scales2)
M2 = R2.T @ S2 @ S2 @ R2
# ic(M2, M)

rot2 = Rotation.from_quat(torch.cat([quat[1:], quat[:1]]))
ic(rot2.as_matrix())
# ic(pytorch3d.transforms.quaternion_to_matrix(quat))
#
# ic(util.build_rotation(quat.reshape(1, -1)))
# ic(pytorch3d.transforms.quaternion_to_matrix(torch.cat([quat[1:], quat[:1]])))
ic("real", R2)

q2 = pytorch3d.transforms.matrix_to_quaternion(R2)
R3 = pytorch3d.transforms.quaternion_to_matrix(quat.reshape(1, 4))
q2 /= torch.linalg.norm(q2)

R4 = quat2mat(q2)
ic(R, R2, R3, R4)
# ic(torch.det(R), torch.det(R2), torch.det(R3))
# print(q2)
# ic(q2)
# ic(quat)
# ic(R2, R3)
M2 = R2.T @ S2 @ S2 @ R2
# ic(M2, M)

covs = contractions.to_cov(scale.reshape(1, 3), quat.reshape(1, 4))
scales3, q3 = contractions.from_covs(M.reshape(1, 3, 3))
covs2 = contractions.to_cov(scales3.reshape(1, 3), q3.reshape(1, 4))
ic(covs, covs2)
ic(scales3, q3)
ic(scale, quat)
