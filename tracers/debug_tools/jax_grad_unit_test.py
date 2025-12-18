import jax
import jax.numpy as jnp
import numpy as np
import slangpy
from pathlib import Path

def jquatToMat3(q):
    r = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    return jnp.array([
                        1.0 - 2.0 * (y * y + z * z),
                        2.0 * (x * y - r * z),
                        2.0 * (x * z + r * y),
                        2.0 * (x * y + r * z),
                        1.0 - 2.0 * (x * x + z * z),
                        2.0 * (y * z - r * x),
                        2.0 * (x * z - r * y),
                        2.0 * (y * z + r * x),
                        1.0 - 2.0 * (x * x + y * y),

    ]).reshape(3, 3).T

fn = slangpy.loadModule(
  str(Path(__file__).parent / "tri_splinetracer/slang/tri-intersect.slang"))
quat = np.array([0.9811850346541486, 0.13331543321474051, 0.07252929475418773, 0.1193416291155902])
rayd = np.ones((3))
scale = np.ones((3))
def test(quat, scale, vec):
    Trayd = jquatToMat3(quat).T @ jnp.diag(1/scale) @ vec
    # Trayd = jquatToMat3(quat).T @  vec
    return (Trayd * Trayd).sum()

def test3(quat, scale, vec):
    Trayd = jquatToMat3(quat).T @ jnp.diag(1/scale) @ vec
    # Trayd = jquatToMat3(quat).T @  vec
    return Trayd

print("\n\nUnit test 3")
vec, dtest3 = jax.vjp(test3, quat, scale, rayd.reshape(-1))
print(vec, dtest3(np.ones((3))))
args = (quat.tolist(), scale.tolist(), rayd.reshape(-1).tolist())
print(fn.test3(*args), fn.dtest3(*args, [1.0, 1.0, 1.0]))

print("\n\nUnit test 1")
vec, dtest = jax.vjp(test, quat, scale, rayd.reshape(-1))
print(vec, dtest(1.0))
print(fn.test(*args), fn.dtest(*args, 1.0))

