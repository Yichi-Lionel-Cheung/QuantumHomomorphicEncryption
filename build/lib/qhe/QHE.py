import qiskit
import numpy as np

from .Dual import Dual_KeyGen, Dual_Enc, Dual_Dec, Dual_Add
from ._common import get_numpy_rng


def init_state(n_qubit, q, B, x, z):
    # create an n-qubit state |0...0> and padded with x,z
    qubits = qiskit.QuantumRegister(n_qubit)
    circuit = qiskit.QuantumCircuit(qubits)
    sk, t_A, A_prime = Dual_KeyGen(n_qubit, q, B)
    ct_x = []
    ct_z = []

    for i in range(n_qubit):
        ct_x.append(Dual_Enc(x[i], A_prime, q, n_qubit, B)[0])
        ct_z.append(Dual_Enc(z[i], A_prime, q, n_qubit, B)[0])
        if x[i] == 1:
            circuit.x(qubits[i])
        if z[i] == 1:
            circuit.z(qubits[i])

    return circuit, sk, t_A, A_prime, ct_x, ct_z


def HE_Cliffod(circuit, gate, index, ct_x, ct_z, q):
    if gate == 'X':
        circuit.x(index)
    if gate == 'Z':
        circuit.z(index)
    if gate == 'H':
        circuit.h(index)
        ct_x[index], ct_z[index] = ct_z[index], ct_x[index]
    if gate == 'S':
        circuit.s(index)
        ct_z[index] = Dual_Add(ct_z[index], ct_x[index], q)
    if gate == 'CNOT':
        circuit.cx(index, index+1)
        ct_z[index[0]] = Dual_Add(ct_z[index[0]], ct_z[index[1]], q)
        ct_x[index[1]] = Dual_Add(ct_x[index[1]], ct_x[index[0]], q)


def CNOT_s(circuit, index, ct_x, ct_z, c_s, sk, A_prime, q, n, B, backend='dummy', seed=None):
    np_rng = get_numpy_rng(seed)
    if backend == 'dummy':
        x = Dual_Dec(ct_x[index], sk, q)
        z = Dual_Dec(ct_z[index], sk, q)
        if x == 1:
            circuit.x(index)
        if z == 1:
            circuit.z(index)
        circuit.s(index)
        ct_z[index] = Dual_Add(ct_z[index], ct_x[index], q)
        # randomly set x and z
        for i in range(2):
            x = np_rng.integers(0, 2)
            z = np_rng.integers(0, 2)
            ct_x[index[i]] = Dual_Enc(x, A_prime, q, n, B)[0]
            ct_z[index[i]] = Dual_Enc(z, A_prime, q, n, B)[0]
            if x == 1:
                circuit.x(index[i])
            if z == 1:
                circuit.z(index[i])
        # TODO: connect to Toffoli

    if backend == 'simulator':
        pass


    if backend == 'qc':
        pass


def HE_Toffoli(circuit, index, ct_x, ct_z, sk, A_prime, q, n, m, B, backend='dummy', seed=None):
    np_rng = get_numpy_rng(seed)
    if backend == 'dummy':
        x = []
        z = []
        for i in index:
            x.append(Dual_Dec(ct_x[i], sk, q))
            z.append(Dual_Dec(ct_z[i], sk, q))
        for i in range(3):
            if x[i] == 1:
                circuit.x(index[i])
            if z[i] == 1:
                circuit.z(index[i])
        circuit.ccx(index[0], index[1], index[2])
        # randomly set x and z
        x = [np_rng.integers(0, 2) for i in range(3)]
        z = [np_rng.integers(0, 2) for i in range(3)]
        for i in range(3):
            ct_x[index[i]] = Dual_Enc(x[i], A_prime, q, n, B)
            ct_z[index[i]] = Dual_Enc(z[i], A_prime, q, n, B)
            if x[i] == 1:
                circuit.x(index[i])
            if z[i] == 1:
                circuit.z(index[i])

    if backend == 'simulator':
        pass

    if backend == 'qc':
        pass
