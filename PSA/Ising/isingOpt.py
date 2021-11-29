
import jax
import jax.numpy as jnp
import numpy as np
from vec2Unitary import vec2Unitary
from jax.config import config
from jax.scipy.linalg import expm
config.update("jax_enable_x64", True)

np.random.seed(1)

def print_matrix(mat):
    with np.printoptions(precision=3, suppress=True):
        print(mat)

def assert_equal(mat1,mat2,eps = 0.0000001):
    assert jnp.sum((mat1-mat2)**2) < eps

# %%
N = 5
beta = 1
J = 0.5
h = 0.2
layer_num = 1
uniParas_num = layer_num * (2**(2*N)-1)

sigma_z = jnp.array([[1,0],[0,-1]])
sigma_x = jnp.array([[0,1],[1,0]])
sigma_y = jnp.array([[0,1],[1,0]])

spin = jnp.array([0,1])
spin_up = jnp.array([[1],[0]])
spin_down= jnp.array([[0],[1]])


# %%
#def createH(J,h)

H = jnp.zeros((2**N,2**N))

initLattice = jnp.kron(sigma_z,sigma_z)
for i in range(2,N):
    initLattice = jnp.kron(initLattice,jnp.eye(2))

H += - J * initLattice

for lattice_point in range(1,N-1):
    curr = jnp.eye(2)
    for i in range(1,lattice_point):
        curr = jnp.kron(curr,jnp.eye(2))
    curr = jnp.kron( jnp.kron(curr,sigma_z),sigma_z)
    for i in range(lattice_point+2,N):
        curr = jnp.kron(curr,jnp.eye(2))
    
    assert curr.shape[0] == H.shape[0]
    
    H += -J * curr


initLattice = sigma_x
for i in range(1,N):
    initLattice = jnp.kron(initLattice,jnp.eye(2))

H += - h * initLattice

for lattice_point in range(1,N-1):
    curr = jnp.eye(2)
    for i in range(1,lattice_point):
        curr = jnp.kron(curr,jnp.eye(2))
    curr = jnp.kron(curr,sigma_x)
    for i in range(lattice_point+1,N):
        curr = jnp.kron(curr,jnp.eye(2))
    
    assert curr.shape[0] == H.shape[0]
    
    H += -h * curr

H = jnp.array(H,dtype=jnp.complex128)

# %%
# prepare the ensemble basis

formator = '{0:' + '0' + str(N)  +'b}'

state_in_str = [formator.format(i) for i in range(2**N)]


def state_to_vec(s):
    # return a probability with the corresponding state
    if s[0] == '1':
        state = spin_up
        state_mat = spin_up
    else:
        state = spin_down
        state_mat = spin_down
    
    for curr in s[1:]:
        if curr == '1':
            state = jnp.kron(state,spin_up)
            state_mat = jnp.hstack((state_mat,spin_up))
        else:
            state = jnp.kron(state,spin_down)
            state_mat = jnp.hstack((state_mat,spin_down))
    
    return state,state_mat

allstate = jnp.stack([state_to_vec(s)[0] for s in state_in_str])
allstateMat = jnp.stack([state_to_vec(s)[1] for s in state_in_str])


# %%
# Utilities

@jax.jit
def stateMat_to_prob(state_mat,probs):
    prob_mat = jnp.multiply(probs,state_mat)
    prob_state = prob_mat[0,:] + prob_mat[1,:]
    return jnp.prod(prob_state)

@jax.jit
def build_ensemble(px):
    single = lambda p,state: p * jnp.outer(state,state)
    mats = jax.vmap(single,(0,0),0)(px,allstate)
    return jnp.sum(mats, axis=0)

@jax.jit
def weighted_expected(operator,px):
    single = lambda state: state.T @ operator @ state
    expectedVs =  jax.vmap(single)(allstate)
    res = jnp.sum(jnp.multiply(expectedVs[:,0,0],px))
    return jnp.real(res)

@jax.jit
def mat_prod(carry,x):
    return carry @ x ,0

@jax.jit
def unitary_prods(uniVec):
    uniVec = uniVec.reshape((layer_num,(2**(2*N)-1)))
    unis = jax.vmap(vec2Unitary,in_axes=(None,0))(N,uniVec)

    res,_ = jax.lax.scan(mat_prod,jnp.eye(2**N,dtype=jnp.complex128),unis)
    return res


stateMat_to_prob_map = jax.vmap(stateMat_to_prob,(0,None),0)

# %% [markdown]
# Free parameters initialization:

# %%
eps = jnp.array(0.5 * np.random.rand(N))
uniVec = jnp.array(np.random.rand(uniParas_num))
paras = jnp.hstack((eps,uniVec))

# %%
def ising(paras):
    eps = paras[:N]
    uniVec = paras[N:]

    p_down = 1/(1+jnp.exp(-beta * eps))
    p_up = jnp.exp(-beta * eps)/(1+jnp.exp(-beta * eps))
    probs = jnp.vstack((p_down,p_up))

    px = stateMat_to_prob_map(allstateMat,probs)
    unitary = unitary_prods(uniVec)
    operator = unitary.conjugate().T @ H @ unitary

    entropy = jnp.sum(jnp.multiply(px,jnp.log(px)))
    loss = entropy + beta* weighted_expected(operator,px)
    #print(entropy)
    return loss




ising(paras),jax.grad(ising)(paras).shape
#ising(paras)



# %%
import scipy.optimize

def value_and_grad_numpy(f):
    def val_grad_f(*args):
        value, grad = jax.value_and_grad(f)(*args)
        return np.float64(value), np.array(grad,dtype=np.float64)
    return val_grad_f
results = scipy.optimize.minimize(value_and_grad_numpy(ising), np.array(paras,dtype=np.float64),
                                  method='L-BFGS-B', jac=True)
print("success:", results.success, "\nniterations:", results.nit, "\nfinal loss:", results.fun)

# %%
-jnp.log(jnp.trace(expm(- beta * H)))

# %%
def exact_ising(J,h,beta):
    inside1 = jnp.exp(beta * J) * jnp.cosh(beta*h) 
    inside2 = jnp.sqrt(jnp.exp(2*beta * J) * (jnp.sinh(beta*h)**2) + jnp.exp(-2*beta * J))

    return - (1/beta) * jnp.log(inside1+inside2)

# %%

p_down = 1/(1+jnp.exp(-beta * eps))
p_up = jnp.exp(-beta * eps)/(1+jnp.exp(-beta * eps))
probs = jnp.vstack((p_down,p_up))
px = stateMat_to_prob_map(allstateMat,probs)
unitary = unitary_prods(uniVec)
operator = unitary.conjugate().T @ H @ unitary

entropy = jnp.sum(jnp.multiply(px,jnp.log(px)))
#print_matrix(unitary.conjugate().T @ unitary)
#operator[2:4,2:4]
beta* weighted_expected(operator,px)
single = lambda state: state.T @ operator @ state
expectedVs =  jax.vmap(single)(allstate)
res = jnp.sum(jnp.multiply(expectedVs[:,0,0],px))
#print(expectedVs)r
res


# %%



