import numpy as np
from scipy.linalg import fractional_matrix_power
from qiskit import *
from qiskit.tools.monitor import job_monitor
#import qiskit.tools.jupyter
from qiskit.tools.visualization import circuit_drawer, plot_histogram
import copy

sigma_z = np.array([[1,0],[0,-1]])
sigma_x = np.array([[0,1],[1,0]])
sigma_y = np.array([[0,1],[1,0]])

spin = np.array([0,1])
spin_up = np.array([[1],[0]])
spin_down= np.array([[0],[1]])



class QuGateWithGrad:
    rDict = {'Rx': 0.5, 'Ry':0.5, 'Rz':0.5, 'Rxx': 0.5, 'Ryy': 0.5,'Rzz': 0.5}

    qiskitGateDict = {'Rx':QuantumCircuit.rx, 'Ry':QuantumCircuit.ry,'Rz':QuantumCircuit.rz,
    'Rxx':QuantumCircuit.rxx,'Ryy':QuantumCircuit.ryy,'Rzz':QuantumCircuit.rzz}

    def __init__(self,symbol: str, registerNum: tuple):
        self.qiskitGate = self.qiskitGateDict[symbol]
        self.r = self.rDict[symbol]
        
        self.registerNum = registerNum
        self.symbol = symbol
        self.gradNum = 0 # gradNum -1/+1 add or minus r
    
    
    def put(self,theta) -> tuple:
        
        return self.qiskitGate, theta + self.gradNum*( np.pi/ (4* self.r))
    

    def setGrad(self,gradNum):

        assert gradNum == -1 or gradNum == 1

        self.gradNum = gradNum

class QuCircWithGrad:

    simShots = 1000

    def createStateDict(self,N):
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
        self.stateDict = stateDict
                

    def __init__(self,N : int):
        self.thetas = []
        self.gates = []
        self.regiNums = []
        self.isConstructed = False
        self.qc = None
        self.N  = N
    
    def add_gate(self,symbol: str,registerNum: tuple,newTheta = None):
        '''
        add a gate with a random theta
        '''
        if type(registerNum) is int:
            registerNum = tuple([registerNum])
        
        if not newTheta:
            newTheta = np.random.random() * 2 * np.pi
        
        newGate = QuGateWithGrad(symbol,registerNum)
        self.regiNums.append(registerNum)
        self.gates.append(newGate)
        self.thetas.append(newTheta)
    
    def add_Arbitrary_gate(self,registerNum: tuple):
        '''
        add a combination of gates creating a space for unitary
        '''
        assert len(registerNum) == 2

        for i in range(2):
            self.add_gate('Rx',registerNum[i])
            self.add_gate('Ry',registerNum[i])
            self.add_gate('Rz',registerNum[i])
        
        self.add_gate('Rxx',registerNum)
        self.add_gate('Ryy',registerNum)
        self.add_gate('Rzz',registerNum)

        for i in range(2):
            self.add_gate('Rx',registerNum[i])
            self.add_gate('Ry',registerNum[i])
            self.add_gate('Rz',registerNum[i])




    def genGradCirc(self,gradIdx : int):
        """
        generate two circuit with plus and minus gradient
        """
        newCircPlus = copy.deepcopy(self)
        newCircMinus = copy.deepcopy(self)

        newCircPlus.gates[gradIdx].setGrad(1)
        newCircMinus.gates[gradIdx].setGrad(-1)

        newCircPlus.construct()
        newCircMinus.construct()

        return newCircPlus,newCircMinus
    
    def genGradCircList(self):
        gradCircs = []
        for i in range(self.size()):
            gradCircs.append(self.genGradCirc(i))
        return gradCircs
    
    def getThetaGrad(self):
        """
        return an array, gradients of the corresponding parameters
        """
        r = 0.5
        circList = self.genGradCircList()
        thetaGrad = np.zeros(self.size())

        for i in range(self.size()):
            circPlus, circMinus = circList[i]
            grad1 = r * (circPlus.calc() - circMinus.calc())
            assert np.abs(np.imag(grad1)) < 0.001
            thetaGrad[i] = grad1

        return thetaGrad
            

    def construct(self):
        '''
        construct the circuit using added gates and append measure
        '''
        N = self.N
        self.isConstructed = True

        self.createStateDict(N)

        self.q = QuantumRegister(N)
        self.c = ClassicalRegister(N)
        self.qc = QuantumCircuit(self.q,self.c)

        for i in range(len(self.gates)):
            gate = self.gates[i]
            theta = self.thetas[i]
            registerNum = self.regiNums[i]
            kitGate, para = gate.put(theta)
            kitGate(self.qc,para,*registerNum)
        
        self.qc.measure(self.q,self.c)
    
    def getCircuit(self):
        assert self.isConstructed, "Circuit not constructed"
        return self.qc
    
    def size(self):
        """
        return the number of gates in the circuit
        """
        return len(self.gates)

    def updateThetas(self,newThetas):
        self.thetas = newThetas
        self.construct()
    
    def setCostFun(self,f):
        """
        set a cost function that have:
            input from the result of a circuit
            output a scalar
        """
        self.costFun = f
    
    def calc(self):

        backend = Aer.get_backend('statevector_simulator')
        job = execute(self.qc, backend, shots=self.simShots)
        res = job.result().get_counts(self.qc)
        cost = self.costFun(res)
        
        return cost


class oneMoreCirc(QuCircWithGrad):

    def construct(self):

        self.isConstructed = True
        self.createStateDict(self.N+1)

        self.q = QuantumRegister(self.N+1)
        self.c = ClassicalRegister(self.N+1)
        self.qc = QuantumCircuit(self.q,self.c)

        self.qc.x(-1)

        for i in range(len(self.gates)):
            gate = self.gates[i]
            theta = self.thetas[i]
            registerNum = self.regiNums[i]
            kitGate, para = gate.put(theta)
            kitGate(self.qc,para,*registerNum)
        
        self.qc.measure(self.q,self.c)


    def setHamiltonian(self,H):
        I = np.identity(H.shape[0])
        newH = np.kron(H,sigma_z) + np.kron(fractional_matrix_power(I - (H @ H),0.5),sigma_x)
        self.H = newH
    
    def setBeta(self,beta):
        self.beta = beta
    
    def calc(self):
        backend = Aer.get_backend('statevector_simulator')
        job = execute(self.qc, backend, shots=self.simShots)
        QCresult = job.result().get_counts(self.qc)
        estF = 0
        for key in QCresult.keys():
            state = self.stateDict[key]
            prob = QCresult[key] / QuCircWithGrad.simShots
            estF += prob * (np.log(prob) + self.beta * state.T @ self.H @ state)
        return estF[0][0]
