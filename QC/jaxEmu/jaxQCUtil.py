import jax
import jax.numpy as jnp
import numpy as np
from jax.config import config
from jax.scipy.linalg import expm
config.update("jax_enable_x64", True)

sigma_z = jnp.array([[1,0],[0,-1]])
sigma_x = jnp.array([[0,1],[1,0]])
sigma_y = jnp.array([[0,1],[1,0]])

spin = jnp.array([0,1])
spin_up = jnp.array([[1],[0]])
spin_down= jnp.array([[0],[1]])

def print_matrix(mat):
    with np.printoptions(precision=3, suppress=True):
        print(mat)

def assert_equal(mat1,mat2,eps = 0.0000001):
    assert jnp.sum((mat1-mat2)**2) < eps

@jax.jit
def Rx(theta):
    return expm(- 1j * theta * sigma_x)

@jax.jit
def Ry(theta: jnp.float64):
    return expm(- 1j * theta * sigma_y)

@jax.jit
def Rz(theta: jnp.float64):
    return expm(- 1j * theta * sigma_z)


