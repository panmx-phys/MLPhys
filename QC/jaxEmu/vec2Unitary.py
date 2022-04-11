import jax
from jax._src.dtypes import dtype
import jax.numpy as jnp

#@jax.jit
def vec2Unitary(N,input):
    '''
    Converts unitary matrix to a vector of parameters (which parameterize the unitary). This vector effectively parameterizes SU(4).


    Args:
        N(int): lattice size
        input(float arrary): 2^(2N)-1 size free parameters

    Returns:
        A complex (2^N,2^N) array that parameterized by the input vector
    '''
    mat_size = 2**N

    # make sure the trace is 0
    vec = jnp.append(input,-jnp.sum(input[-mat_size+1:]))

    h_mat = jnp.diag(vec[-mat_size:])
    


    count = 0
    #real part
    for i in range(1,mat_size):
        h_mat += jnp.diag(vec[count:count+i],k=-mat_size+i)
        count += i

    #img part
    for i in range(1,mat_size):
        h_mat += jnp.diag(1j * vec[count:count+i],k=-mat_size+i)
        count += i

    h_mat = (h_mat.conjugate().T + h_mat)/2
    unitary = jax.scipy.linalg.expm(1j * h_mat)
    return unitary