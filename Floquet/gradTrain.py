# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import jax.numpy as jnp
from jax import grad, jit, vmap, ops
from jax import random
import numpy as np
import jax
from jax.scipy.linalg import expm
import eigAD
from jax.config import config   
import scipy.optimize
config.update("jax_enable_x64", True)


# %%
# tunning parameter (constant in grad)
numStates=25
numSteps=31
numBands=5
fr=25.18
kpoints=250
kvec=jnp.linspace(-1,0,kpoints)

def tridiag(a, b, c, k1=-1, k2=0, k3=1):
    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)


C = []
for i in range(2,numStates,2):
    C = C + [0,i]
C = np.array(C)
D = np.zeros(numStates)
M1 = tridiag(C,D,-1 * C)

E = [0]
for i in range(2,numStates,2):
    E = E + [(i)**2,(i)**2]
M2 = np.array(-np.diag(E))


F = np.concatenate((np.array([np.sqrt(2)]), np.ones(numStates-3)))
M3 = np.diag(F,-2) + np.diag(F,2)


# %%
freq = 70.0
Ttot = 1/freq
ftot = 1/Ttot
dT = Ttot/numSteps
N = 31

tVec = jnp.linspace(0,Ttot,N)
dT = tVec[1] - tVec[0]
tVec = tVec[:-1]
tVec = tVec + dT/2
modfunc = 1 + 0.5 * jnp.sin(2 * jnp.pi * freq * tVec)


# %%
# freqs: driving frequencies
# alphas: driving strength

M1 = jnp.asarray(M1,dtype=jnp.complex64)
M2 = jnp.asarray(M2,dtype=jnp.complex64)
M3 = jnp.asarray(M3,dtype=jnp.complex64)

def computeFloquetLoss(modulation):
    
    A = modulation[0]
    modulation = modulation[1:]
    dTau = (2 * jnp.pi * fr) * dT

    @jax.jit
    def perKstep(k):
        unitaryInit = jnp.identity(M1.shape[0],dtype=jnp.complex64)

        @jax.jit
        def scanf(unitary,tIdx):
            dU = expm(-1j * dTau * createHmat(tIdx,k))
            dU = jnp.asarray(dU,dtype=jnp.complex64)
            unitary = jnp.asarray(unitary,dtype=jnp.complex64)
            unitary = jnp.matmul(unitary,dU)
            return unitary,0

        res, _ = jax.lax.scan(scanf,unitaryInit,jnp.arange(N))
        return res


    @jax.jit
    def createHmat(tIdx,k):
        newMat = (k**2) * jnp.identity(numStates,dtype=jnp.complex64) - 2* 1j * k * M1 -M2  - (1/4) * M3 * A * modulation[tIdx]
        return newMat
    

    def genUni():
        kMap = vmap(perKstep)
        return kMap(kvec)

    res = genUni()

    def eigWrapper(mat):
        return eigAD.eig(mat)

    eigWrapper= jax.jit(eigWrapper,backend='cpu')
    eigWrapper= vmap(eigWrapper)
    b,vF = eigWrapper(res)
    rawEfloquet = jnp.real(1j*jnp.log(b)* (ftot/fr) / (2*np.pi))
    
    @jax.jit
    def blochStates(i):
        k = kvec[i]
        currF = vF[i,:,:] 
        H0 =  (k**2) * jnp.identity(numStates) - 2* 1j * k * M1 -M2  - (1/4) * M3 * A 
        a,vS = jnp.linalg.eigh(H0)
        vS = jnp.transpose(vS)
        Cvec = jnp.matmul(vS,jnp.conjugate(currF))
        Pvec = jnp.multiply(Cvec,jnp.conjugate(Cvec))
        inds = jnp.argmax(jnp.real(Pvec),axis=1)
        Efloquet = rawEfloquet[i,inds[:numBands]]
        return Efloquet
    bandsF = vmap(blochStates)(jnp.arange(250))

    return jnp.std(bandsF[:,0])
    
    


# %%
init_para = np.ones(N)
init_para[0] = 2.0
init_para[1:] = modfunc

res = init_para
def callback(xk):
    global res
    res = np.vstack((res,xk))
    np.save("/data/hpan/trivial.npy",res)



def value_and_grad_numpy(f):
    def val_grad_f(*args):
        value, grad = jax.value_and_grad(f)(*args)
        return np.array(value), np.array(grad)
    return val_grad_f
results = scipy.optimize.minimize(value_and_grad_numpy(computeFloquetLoss), np.array(init_para),
                                  method="L-BFGS-B", jac=True,callback=callback)
print("success:", results.success, "\nniterations:", results.nit, "\nfinal loss:", results.fun)



