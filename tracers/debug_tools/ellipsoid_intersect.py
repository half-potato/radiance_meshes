
import torch
import slangtorch
from pathlib import Path
from icecream import ic
kernels = slangtorch.loadModule(
    str(Path(__file__).parent.parent / "ellipsoid_splinetracer/slang/tri-intersect.slang")
)

def l2_normalize_th(x, eps=torch.finfo(torch.float32).eps):
    """Normalize x to unit length along last axis."""
    return x / torch.sqrt(
        torch.clip(torch.sum(x**2, dim=-1, keepdim=True), eps, None)
    )

def dot(a, b):
    return (a*b).sum()

def rotate_vector(v, q):
    t = 2 * torch.cross(-q[1:], v);
    return v + q[0] * t + torch.cross(-q[1:], t);

def eliIntersect( ro, rd, ra ):
    radius = 1;
    ra_norm = ra / radius;
    ocn = ro/ra_norm;
    rdn = rd/ra_norm;
    a = dot( rdn, rdn );
    bp = -dot( ocn, rdn );
    l = (ocn + bp / a * rdn);
    h = a*(radius*radius - dot(l, l));
    c = dot( ocn, ocn ) - radius*radius;
    if ( h<0.0 ):
        return torch.tensor([-1, -1])
    q = bp + torch.sign(bp) * torch.sqrt(h);
    return [c/q, q/a]

def ray_intersect_ellipsoid( rayo, rayd, scales, mean, quat):
    Trayd = rotate_vector(rayd, quat)
    Trayo = rotate_vector(rayo - mean, quat)

    fminmaxt = eliIntersect(Trayo, Trayd, scales);
    return [min(fminmaxt[0], fminmaxt[1]), max(fminmaxt[0], fminmaxt[1])]

rayd = torch.tensor([0.0, 0, 1])
mean = torch.tensor([0.0, 0, 1])
quat = l2_normalize_th(2*torch.rand(4)-1)

for s in torch.linspace(1e-5, 1e-1, 10):
    # scales = s*torch.tensor([1.5, 0.1, 0.1])
    scales = s*torch.tensor([1.0, 1.0, 1.0])
    max_err = 0
    for s in torch.linspace(0, scales.max(), 1000):
        rayo = torch.tensor([0.0, s, 0])
        t0, t1 = kernels.ray_intersect_ellipsoid_th(rayo.tolist(), rayd.tolist(), scales.tolist(), mean.tolist(), quat.tolist())
        dtype = torch.DoubleTensor
        l0, l1 = ray_intersect_ellipsoid( rayo.type(dtype), rayd.type(dtype), scales.type(dtype), mean.type(dtype), quat.type(dtype))
        if l0 < 0:
            continue
        err0 = abs(t0 - l0)
        err1 = abs(t1 - l1)
        err = (err0 + err1)/2
        if err > 1:
            print(l0, t0, l1, t1)
        max_err = max(max_err, err)
    print(max_err)
