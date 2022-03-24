
from QCutils import *
import numpy as np
import numpy as jnp
from scipy.linalg import expm
import scipy.optimize



N = 6
J = 0.5
h = 0.5


def QCCalc(beta,J,h,layerNum):
    sigma_z = np.array([[1,0],[0,-1]])
    sigma_x = np.array([[0,1],[1,0]])
    sigma_y = np.array([[0,1],[1,0]])

    spin = np.array([0,1])
    spin_up = np.array([[1],[0]])
    spin_down= np.array([[0],[1]])


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


    formator = '{0:' + '0' + str(N)  +'b}'

    state_in_str = [formator.format(i) for i in range(2**N)]

    def state_to_vec(s):
        
        if s[0] == '1':
            state = spin_up
            state_mat = spin_up
        else:
            state = spin_down
            state_mat = spin_down
        
        for curr in s[1:]:
            if curr == '1':
                state = np.kron(state,spin_up)
                state_mat = np.hstack((state_mat,spin_up))
            else:
                state = np.kron(state,spin_down)
                state_mat = np.hstack((state_mat,spin_down))
        
        return state,state_mat

    stateDict = dict()
    for s in state_in_str:
        stateDict[s] = state_to_vec(s)[0]




    def FCostFunc(QCresult):
        estF = 0
        for key in QCresult.keys():
            state = stateDict[key]
            prob = QCresult[key] / QuCircWithGrad.simShots
            estF += prob * (np.log(prob) + beta * state.T @ H @ state)
        return estF[0][0]


    test1 = QuCircWithGrad(N)

    for i in range(N):
        test1.add_gate('Rz',i)

    for layer in range(layerNum):
        for i in range(N-1):
            test1.add_gate('Rxx',(i,i+1))
            test1.add_gate('Ryy',(i,i+1))
            test1.add_gate('Rzz',(i,i+1))

        for j in range(N-1):
            i = (N-2 - j)
            test1.add_gate('Rxx',(i,i+1))
            test1.add_gate('Ryy',(i,i+1))
            test1.add_gate('Rzz',(i,i+1))


    test1.setCostFun(FCostFunc)
    test1.construct()


    def ValandGrad(thetas):
        test1.updateThetas(thetas)
        return test1.calc(), test1.getThetaGrad()



    initThetas = 2* np.pi * np.random.rand(test1.size())
    results = scipy.optimize.minimize(ValandGrad, initThetas,
                                    method='L-BFGS-B', jac=True)
    return results.fun,float(-np.log(np.trace(expm(- beta * H))))


def plainCalc(beta,J,h,layerNum):
    sigma_z = np.array([[1,0],[0,-1]])
    sigma_x = np.array([[0,1],[1,0]])
    sigma_y = np.array([[0,1],[1,0]])

    spin = np.array([0,1])
    spin_up = np.array([[1],[0]])
    spin_down= np.array([[0],[1]])


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



    return float(-np.log(np.trace(expm(- beta * H))))


    


