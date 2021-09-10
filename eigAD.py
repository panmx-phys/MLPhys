from jax import custom_vjp
from jax._src.numpy.lax_numpy import asarray
import jax.numpy as jnp
import jax

@custom_vjp
def eig(mat):
    return jnp.linalg.eig(mat)


def eig_fwd(mat):
    res = jnp.linalg.eig(mat)
    return res,res

def eig_bwd(res, g):
    e,u = res
    n = e.shape[0]
    ge, gu = g
    ge = jnp.diag(ge)

    f = e[..., jnp.newaxis, :] - e[..., :, jnp.newaxis] + 1.e-20
    diag_elements = jnp.diag_indices_from(f)
    f = jax.ops.index_update(f, diag_elements, jnp.inf)
    f= 1./f

    #f = 1/(e[..., jnp.newaxis, :] - e[..., :, jnp.newaxis] + 1.e-20)
    #f -= jnp.diag(f)


    ut = jnp.swapaxes(u, -1, -2)
    r1 = f * jnp.dot(ut, gu)
    r2 = -f * (jnp.dot(jnp.dot(ut, jnp.conj(u)), jnp.real(jnp.dot(ut,gu)) * jnp.eye(n)))
    r = jnp.dot(jnp.dot(jnp.linalg.inv(ut), ge + r1 + r2), ut)
    return (jnp.asarray(r,dtype=jnp.complex64),)

eig.defvjp(eig_fwd, eig_bwd)