from jax import config
config.update("jax_enable_x64", True)
import jax_render
import jax
import jax.numpy as jnp
import numpy as np
from numpy import array
import util
import quadrature
from jax.test_util import check_grads
from icecream import ic

key = 'scale'
target_out = np.ones((3))
tri_ids = [2, 18, 15, 7]
eps = 1e-5

def query_l0( tdist, rayo, rayd, params, eps=jnp.finfo(jnp.float64).eps):
    xs = rayo.reshape(1, 3) + tdist.reshape(-1, 1) * rayd.reshape(1, 3)
    R = util.jquatToMat3(util.jl2_normalize(params['quat']))
    sc = (((xs - params['mean'].reshape(1, 3)) @ R.T) / jnp.maximum(params['scale'].reshape(1, 3), 1e-8))
    d = jnp.linalg.norm(sc, axis=-1, ord=1)
    densities = jnp.where(d < 1, params['density'], 0)
    colors = jnp.where(d < 1, params['feature'].reshape(1, 3), 0)
    return densities, colors

def query_l1( tdist, rayo, rayd, params, eps=jnp.finfo(jnp.float64).eps):
    xs = rayo.reshape(1, 3) + tdist.reshape(-1, 1) * rayd.reshape(1, 3)
    R = util.jquatToMat3(util.jl2_normalize(params['quat']))
    # R = util.jquatToMat3(params['quat'])
    sc = (((xs - params['mean'].reshape(1, 3)) @ R.T) / jnp.maximum(params['scale'].reshape(1, 3), 1e-8))
    d = jnp.linalg.norm(sc, axis=-1, ord=1)
    densities = params['density'] * jnp.clip(1-d, 0, None)
    colors = params['feature']

    return densities, colors.reshape(1, 3)

t_range = (0, 12)
num_quad = 2**24
tdist = jnp.linspace(*t_range, num_quad + 1)
kernel = query_l1

def quad_l2_loss(params, rayo, rayd):
    quad_color, quad_extras = quadrature.render_quadrature(
        tdist,
        lambda t: kernel(t, rayo, rayd, params),
        return_extras=True,
    )
    return ((quad_color - target_out)**2).mean()

def l2_loss(params, rayo, rayd):
    color = jax_render.render(params, rayo, rayd, tri_ids)
    return ((color - target_out)**2).mean()

params1 = {#'scale': np.array([0.1, 0.1, 0.1]),
           'scale': np.array([0.1       , 0.15848933, 0.25118864]),
          'mean': np.array([0.   , 0.002, 1.   ]),
          'quat': np.array([0.45220585, 0.58929456, 0.49665892, 0.44896738]),
          'density': np.array([4.66073528]),
          'feature': np.array([0.525 , 0.525 , 0.7311]),
}


params2 = {'scale': params1['scale'] - array([0.0, 0.0, 0.0+eps], dtype=np.float64),
# params2 = {'scale': params1['scale'] - array([0.0+eps, 0.0+eps, 0.0+eps], dtype=np.float64),
          # 'mean': array([0.   , 0.002, 1.   ], dtype=np.float64),
          'mean': params1['mean'],
          'quat': params1['quat'],# + np.array([eps, 0, 0, 0]),
          # 'quat': array([0.45220587, 0.5892946 , 0.49665895, 0.4489674 ], dtype=np.float64),
          'density': params1['density'],
          # 'feature': array([0.5249792 , 0.5249792 , 0.73105854], dtype=np.float64)
          'feature': params1['feature'],
}

params1 = {k: v.astype(np.float64) for k, v in params1.items()}
params2 = {k: v.astype(np.float64) for k, v in params2.items()}

rayo = np.array([[0., 0., 0.]])
rayd = np.array([[-0.00436327, -0.00436327,  0.99998096]])
# rayd = np.array([[-0.00414484, -0.00436327,  0.99998189]])

grad1 = jax.grad(l2_loss)(params1, rayo, rayd)
grad2 = jax.grad(l2_loss)(params2, rayo, rayd)

grad3 = jax.grad(quad_l2_loss)(params1, rayo, rayd)
grad4 = jax.grad(quad_l2_loss)(params2, rayo, rayd)

color1 = jax_render.render(params1, rayo, rayd, tri_ids)
color2 = jax_render.render(params2, rayo, rayd, tri_ids)

quad_color1, quad_extras = quadrature.render_quadrature(
    tdist,
    lambda t: kernel(t, rayo, rayd, params1),
    return_extras=True,
)

print(f'grad1[{key}]', grad1[key])
print(f'grad3[{key}]', grad3[key])
print(f'grad2[{key}]', grad2[key])
print(f'grad4[{key}]', grad4[key])

quad_color2, quad_extras = quadrature.render_quadrature(
    tdist,
    lambda t: kernel(t, rayo, rayd, params2),
    return_extras=True,
)

print(key, grad4[key].dtype)
print(color1.dtype, color2.dtype)
print(quad_color1.dtype, quad_color2.dtype)

color_vec = 2 * (target_out - quad_color1) / 3

print(((color1 - color2) / eps) @ color_vec)
print(((quad_color1 - quad_color2) / eps) @ color_vec)
ic(color1, color2)
ic(quad_color1, quad_color2)

for key in ['mean', 'scale', 'quat', 'density', 'feature']:
    print(f'grad1[{key}]', grad1[key])
    print(f'grad3[{key}]', grad3[key])
    print(f'grad2[{key}]', grad2[key])
    print(f'grad4[{key}]', grad4[key])

check_grads(l2_loss, (params1, rayo, rayd), order=1, modes='rev', eps=1e-4)#, atol=2e-3, rtol=1e-2)
check_grads(quad_l2_loss, (params1, rayo, rayd), order=1, modes='rev', atol=1e-3, rtol=1e-4)
print("Both pass")
