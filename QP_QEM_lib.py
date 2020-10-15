import openfermion as of
import openfermioncirq as ofc
import cirq
import numpy as np
import itertools as it
import math
import json

from scipy.sparse.linalg import expm, svds
from scipy import sparse
from typing import Sequence, Tuple, Dict, Any
from tqdm import tqdm


class PhaseS(cirq.SingleQubitGate):

    def __init__(self, exponent=1):
        self.exponent = exponent

    def _unitary_(self):
        unitary = (1/np.sqrt(2))*(cirq.I._unitary_() - 1j*cirq.Z._unitary_())

        return np.linalg.matrix_power(unitary, self.exponent)

    def __str__(self):
        return 'S**{}'.format(self.exponent)


class Rx(cirq.SingleQubitGate):

    def __init__(self, exponent=1):
        self.exponent = exponent

    def _unitary_(self):
        unitary = cirq.H._unitary_() @ (np.linalg.matrix_power(PhaseS()._unitary_(), 3) @ cirq.H._unitary_())

        return np.linalg.matrix_power(unitary, self.exponent)

    def __str__(self):
        return 'Rx**{}'.format(self.exponent)


class Ry(cirq.SingleQubitGate):

    def __init__(self, exponent=1):
        self.exponent = exponent

    def _unitary_(self):
        unitary = Rz(3)._unitary_() @ (Rx(1)._unitary_() @ Rz(1)._unitary_())

        return np.linalg.matrix_power(unitary, self.exponent)

    def __str__(self):
        return 'Ry**{}'.format(self.exponent)


class Rz(cirq.SingleQubitGate):

    def __init__(self, exponent=1):
        self.exponent = exponent

    def _unitary_(self):
        unitary = np.linalg.matrix_power(PhaseS()._unitary_(), 3)

        return np.linalg.matrix_power(unitary, self.exponent)

    def __str__(self):
        return 'Rz**{}'.format(self.exponent)


class Pz(cirq.SingleQubitGate):

    def _unitary_(self):
        return (1/2)*(cirq.I._unitary_() + cirq.Z._unitary_())

    def __str__(self):

        return 'P'+'\u2083'


class TwoQubitDepolarizingChannel(cirq.ops.gate_features.TwoQubitGate):
    '''
    Two qubit depolarizing channel based on single qubit depolarizing channel from Cirq

    '''

    def __init__(self, p: float) -> None:

        self._p = cirq.value.validate_probability(p/15, 'p')
        self._p_i = 1 - cirq.value.validate_probability(16*p/15, 'sum_p')

    def _mixture_(self) -> Sequence[Tuple[float, np.ndarray]]:

        tuple_2q = ((self._p_i, np.kron(cirq.protocols.unitary(cirq.ops.identity.I),
                                        cirq.protocols.unitary(cirq.ops.identity.I))),)

        for op_tup in it.product([cirq.protocols.unitary(cirq.ops.identity.I),
                                  cirq.protocols.unitary(
                                      cirq.ops.pauli_gates.X),
                                  cirq.protocols.unitary(
                                      cirq.ops.pauli_gates.Y),
                                  cirq.protocols.unitary(cirq.ops.pauli_gates.Z)], repeat=2):

            tuple_2q += ((self._p, np.kron(*op_tup)),)

        return tuple_2q

    def _has_mixture_(self) -> bool:
        return True

    def _value_equality_values_(self):
        return self._p

    def __repr__(self) -> str:
        return 'cirq.two_qubit_depolarize(p={!r})'.format(
            self._p
        )

    def __str__(self) -> str:
        return 'two_qubit_depolarize(p={!r})'.format(
            self._p
        )

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs'
                               ) -> 'cirq.CircuitDiagramInfo':
        return cirq.protocols.CircuitDiagramInfo(
            wire_symbols=('D({:.3f})'.format(self._p), 'D({:.3f})'.format(self._p)))

    @property
    def p_(self) -> float:
        """The probability that an error occurs."""
        return self._p

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['p'])


def two_qubit_depolarize(p: float) -> TwoQubitDepolarizingChannel:
    '''
    Wrapper for TwoQubitDepolarizingChannel. Follows closely Cirq style for single qubit version.

    '''

    return TwoQubitDepolarizingChannel(p)


def multi_kron(tup):
    '''
    Extension of np.kron() to more than two operators.

    Input:
    tup [tuple] : tuple of operators to use for Kronecker product

    Output:
    A [array] : Kronecker product of operators in tup

    '''

    A = np.eye(1)

    for op in tup:
        A = np.kron(A, op)

    return A


def apply_to_qubit(op, qubit):
    '''
    Apply cirq.Gate or list of cirq.Gates on qubit. Useful for gates composed of multiple cirq.Gates.

    Input:
    op   [cirq.Gate/tuple/list] : gate, tuple of gates or list of gates to be applied on qubit
    qubit [cirq.LineQubit/cirq.GridQubit/cirq.NamedQubit] : qubit to apply op on

    Output:
    op_q [cirq.Gate/tuple/list] : gate, tuple of gates or list of gates applied on qubit

    '''

    op_type = type(op)

    if op_type == list or op_type == tuple or op_type == np.ndarray:
        op_q = op_type(map(lambda x: x.on(qubit), op))

    else:
        op_q = op.on(qubit)

    return op_q


def basis_ops_sim():
    '''
    Generate 16 single qubit operators for decomposition using Ref. 2's definition of Rx,Ry,Rz
    Used for circuit simulation using cirq.Simulator.
    Instead of Pz gate, uses cirq.MeasurementGate for postselection detailed in Ref. 2.

    Output:
    ops [list] : list of basis operations

    Current function not in use, created for debugging purposes only.

    '''

    rx = Rx(1)
    ry = Ry(1)
    rz = Rz(1)

    rx2 = Rx(2)
    ry2 = Ry(2)
    rz2 = Rz(2)

    rx3 = Rx(3)
    ry3 = Ry(3)
    rz3 = Rz(3)

    # I
    I = cirq.I

    ops = []

    ops.append((I,))

    # R_alpha
    ops.extend(((rx2,), (rz2, rx2), (rz2,)))

    # R_a
    ops.extend(((rx,), (ry,), (rz,)))

    # R_ab
    ops.extend(((rz2, rx), (rz, rx, rz), (rz, rx2)))

    # pi_a
    ops.extend(
        ((rz, rx, cirq.MeasurementGate(1, key='M'), rx3, rz3),
         (rx3, cirq.MeasurementGate(1, key='M'), rx),
         (cirq.MeasurementGate(1, key='M'),))
    )

    # pi_ab
    ops.extend(
        ((rz, rx3, cirq.MeasurementGate(1, key='M'), rx3, rz3),
         (rz2, rx3, cirq.MeasurementGate(1, key='M'), rx),
         (rx2, cirq.MeasurementGate(1, key='M')))
    )

    return ops


def basis_ops():
    '''
    Generate 16 single qubit operators for decomposition using Ref. 2's definition of Rx,Ry,Rz
    Used for direct matrix multiplication instead of circuit simulation using cirq.Simulator.
    Pz is used since circuit simulation is performed using matrix multiplication instead of cirq.Simulator.

    Output:
    ops [list] : list of basis operations

    Current function not in use, created for debugging purposes only.

    '''

    rx = Rx(1)
    ry = Ry(1)
    rz = Rz(1)

    rx2 = Rx(2)
    ry2 = Ry(2)
    rz2 = Rz(2)

    rx3 = Rx(3)
    ry3 = Ry(3)
    rz3 = Rz(3)

    # I
    I = cirq.I

    ops = []

    ops.append((I,))

    # R_alpha
    ops.extend(((rx2,), (rz2, rx2), (rz2,)))

    # R_a
    ops.extend(((rx,), (ry,), (rz,)))

    # R_ab
    ops.extend(((rz2, rx), (rz, rx, rz), (rz, rx2)))

    # pi_a
    ops.extend(
        ((rz, rx, Pz(), rx3, rz3),
         (rx3, Pz(), rx),
         (Pz(),))
    )

    # pi_ab
    ops.extend(
        ((rz, rx3, Pz(), rx3, rz3),
         (rz2, rx3, Pz(), rx),
         (rx2, Pz()))
    )

    return ops


def noise_model(dim, error_type, error):
    '''
    Wrapper function for any relevant noise models we can consider. 
    Used when simulation is performed by running circuit using cirq.Simulator instead of matrix
    multiplication.

    Input:
    rho        [array] : array on which to apply noise channel
    error_type   [str] : noise model chosen
    error [float/list] : float or list for probability/aplitude of noise

    Output:
    rho_n [array] : rho after applying noise channel

    '''

    if error_type == None:
        return cirq.I

    if error_type == 'depolarize':
        if type(error) != list and type(error) != tuple and type(error) != np.ndarray:

            if dim == 1:
                return cirq.depolarize(error)

            elif dim == 2:
                return two_qubit_depolarize(error)

        else:
            return cirq.asymmetric_depolarize(*error)

    elif error_type == 'amp_damp':
        return cirq.amplitude_damp(error)

    elif error_type != 'depolarize' and error_type != 'amp_damp':
        raise TypeError("Noise model not recognized")


def noise_model_channel(rho, error_type, error):
    '''
    Apply channel to density matrix any noise model we can consider (currently only for depolarizing noise).
    Useful when consider matrix multiplication for circuit simulation instead of measurements.

    Input:
    rho        [array] : array on which to apply noise channel
    error_type   [str] : noise model chosen
    error [float/list] : float or list for probability/aplitude of noise

    Output:
    rho_n [array] : rho after applying noise channel

    '''

    if error_type == 'depolarize':
        pauli_ops = [multi_kron(p) for p in it.product([cirq.I._unitary_(), cirq.X._unitary_(), cirq.Y._unitary_(), cirq.Z._unitary_()], repeat=int(math.log(len(rho), 2)))]

        # if symmetric depolarization is chosen
        if type(error) != list and type(error) != tuple and type(error) != np.ndarray:
            coeff = error/(len(list(pauli_ops))-1) * np.ones(len(pauli_ops))

            rho_n = (1-len(list(pauli_ops))*coeff[0])*rho
            if sum(coeff[1:]) > 1:
                raise TypeError('px+py+pz>1')

        # if asymmetric depolarization is chosen
        else:
            error = np.array(error, dtype=float)
            coeff = np.insert(error, 0, sum(error)/(len(list(pauli_ops))-1))
            rho_n = (1-len(list(pauli_ops))*coeff[0])*rho
            if sum(coeff[1:]) > 1:
                raise TypeError('px+py+pz>1')

        for p in range(len(pauli_ops)):
            rho_n = rho_n + coeff[p] * ((pauli_ops[p] @ rho) @ pauli_ops[p])

        return rho_n

    else:
        raise TypeError("Noise model not recognized")


def swap_circuit(n, qubits, decompose=True):
    '''
    Create SWAP circuit in Ref.2.

    Input:
    n        [int] : number of qubits in each register to be swapped
    qubits [tuple] : tuple of qubits on which to define the circuit

    Output:
    circuit_swap [cirq.Circuit] : SWAP circuit

    '''

    assert(len(qubits) == 2*n+1)

    circuit_swap = cirq.Circuit()
    circuit_swap.append([cirq.H(qubits[0]), cirq.H(qubits[1])])

    for j in range(2, n+1):
        circuit_swap.append(cirq.CNOT(qubits[1], qubits[j]))

    for j in range(1, n+1):
        if decompose:
            circuit_swap.append(cirq.CSWAP(
                qubits[0], qubits[j], qubits[j+n])._decompose_())
        else:
            circuit_swap.append(cirq.CSWAP(qubits[0], qubits[j], qubits[j+n]))

    circuit_swap.append(cirq.H(qubits[0]))
    circuit_swap.append(cirq.measure(qubits[0], key='M'))

    return circuit_swap


def noisy_circuit(circuit, error_type=None, error=None):
    '''
    Create noisy circuit for benchmarking QEM results.

    Input:
    circuit [cirq.Circuit] : ideal circuit to apply single and two qubit noise to
    error_type       [str] : noise model chosen
    error     [float/list] : float or list for probability/aplitude of noise

    Output:
    circuit_noisy [cirq.Circuit] : noisy SWAP circuit

    '''

    qubits = circuit.all_qubits()
    ops = circuit.all_operations()

    circuit_noisy = cirq.Circuit()

    for op in ops:
        circuit_noisy.append(op)

        if type(op.gate).__name__ != 'MeasurementGate':
            circuit_noisy.append(noise_model(
                dim=len(op.qubits), error_type=error_type, error=error)(*op.qubits))

    return circuit_noisy


def o_tilde_1q(gate, noisy, error_type, error):
    '''
    PTM representation of a gate. Need to shorten with list comprehensions, but this was mostly as a sanity check
    for the GST method o_tilde_1q_tomo()

    Input:
    gate  [cirq.Gate] : gate to be reconstructed
    noisy [bool]      : implement either noisy or ideal version of PTM 
    error [float]     : probability/amplitude for noisy model dictated by instance of noise_model_channel()

    Output:
    O_q [array] : (4^n, 4^n) PTM for the input gate 

    '''

    try:
        qubit = gate.qubits[0]

    except:
        qubit = gate[0].qubits[0]

    rho_k = []
    O_q = np.zeros((4, 4), dtype=complex)

    Q_ops = [cirq.I(qubit), cirq.X(qubit), cirq.Y(qubit), cirq.Z(qubit)]

    for op in Q_ops:
        rho = op._unitary_()

        if noisy:
            #rho = noise_model_channel(rho, error_type, error)

            if type(gate) == tuple or type(gate) == list:
                for g in gate:
                    circuit = cirq.Circuit(g).unitary()
                    rho = (circuit @ rho) @ (circuit.conj().T)

                rho = noise_model_channel(rho, error_type, error)
                rho = .5*rho

            else:
                circuit = cirq.Circuit(gate).unitary()
                rho = (circuit @ rho) @ (circuit.conj().T)
                rho = .5*noise_model_channel(rho, error_type, error)

        else:
            # operator = cirq.unitary(cirq.I)
            # for uni in gate:
            #     operator = cirq.unitary(uni) @ operator

            circuit = cirq.Circuit(gate).unitary()
            rho = .5*(circuit @ rho) @ (circuit.conj().T)

        rho_k.append(rho)

    for j in range(len(Q_ops)):
        for k in range(len(rho_k)):
            Q_j = Q_ops[j]._unitary_()
            Q_rho = Q_j@rho_k[k]

            # if noisy:
            #    Q_rho = noise_model_channel(Q_rho, error_type, error)

            Ojk = np.trace(Q_rho)
            O_q[j][k] = Ojk

    return O_q


def o_tilde_1q_tomo(gate, repetitions, noisy, error):
    '''
    PTM representation of a gate reconstructed using GST.

    Input:
    gate  [cirq.Gate] : gate to be reconstructed
    repetitions [int] : number of repetitions for tomography
    noisy      [bool] : implement either noisy or ideal version of PTM 
    error     [float] : probability/amplitude for noisy model dictated by instance of noise_model_channel()

    Output:
    O_q [array] : (4^n, 4^n) PTM for the input gate 

    '''

    try:
        qubit = gate.qubits[0]

    except:
        qubit = gate[0].qubits[0]

    if noisy:
        noise = noise_model(dim=1, error_type=error_type, error=error)(qubit)
        if type(gate) == tuple or type(gate) == list:
            gate = [n_g for n_g in zip(gate, [noise]*len(gate))]

        else:
            gate = (gate, noise)

    else:
        noise = noise_model(dim=1, error_type=None, error=error)(qubit)

    rho_k = []
    O_q = np.zeros((4, 4), dtype=complex)

    tomo_ops = [cirq.I(qubit), cirq.X(qubit), cirq.ry(
        np.pi/2)(qubit), cirq.rx(-np.pi/2)(qubit)]
    Q_ops = [cirq.I(qubit), cirq.X(qubit), cirq.Y(qubit), cirq.Z(qubit)]
    U_ops = [cirq.I(qubit), cirq.H(qubit), cirq.rx(
        np.pi/2)(qubit), cirq.I(qubit)]

    for j, k in it.product(range(len(U_ops)), repeat=2):
        circuit = cirq.Circuit(tomo_ops[k], noise, gate, U_ops[j], noise)

        if circuit.has_measurements():
            circuit.append(cirq.measure(qubit, key='M_post'))
            result_k = cirq.Simulator().run(program=circuit, repetitions=repetitions)
            hist_post = result_k.multi_measurement_histogram(
                keys=('M', 'M_post'))

            N_fail = hist_post[(1, 0)]+hist_post[(1, 1)]
            N_0 = hist_post[(0, 0)]
            N_1 = hist_post[(0, 1)]

            if j == 0:
                O_q[j][k] = (N_0+N_1)/repetitions
            else:
                O_q[j][k] = (N_0-N_1)/repetitions

        else:
            circuit.append(cirq.measure(qubit, key='M_post'))
            result_k = cirq.Simulator().run(program=circuit, repetitions=repetitions)

            if j == 0:
                O_q[j][k] = 1

            else:
                p0 = result_k.histogram(key='M_post')[0]/repetitions
                O_q[j][k] = 2*p0-1

    return O_q


def o_tilde_2q(gate, noisy, error_type, error):
    '''
    2 qubit version of o_tilde_1q

    Input:
    gate  [cirq.Gate] : gate to be reconstructed
    noisy [bool]      : implement either noisy or ideal version of PTM 
    error [float]     : probability/amplitude for noisy model dictated by instance of noise_model_channel()

    Output:
    O_q [array] : (4^n, 4^n) PTM for the input gate 

    '''

    try:
        qubits = gate.qubits

    except:
        qubits = (g.qubits[0] for g in gate)

    rho_k = []
    O_q = np.zeros((16, 16), dtype=complex)

    Q_ops = [cirq.I, cirq.X, cirq.Y, cirq.Z]

    for op1, op2 in it.product(Q_ops, repeat=2):
        rho = np.kron(op1._unitary_(), op2._unitary_())

        if noisy:
            #rho = noise_model_channel(rho, error_type, error)

            if type(gate) == tuple or type(gate) == list:
                circuit = cirq.Circuit(gate).unitary()
                rho = (circuit @ rho) @ (circuit.conj().T)
                rho = noise_model_channel(rho, error_type, error)

                rho = (.5**2)*rho

            else:
                circuit = cirq.Circuit(gate).unitary()
                rho = (circuit @ rho) @ (circuit.conj().T)
                rho = (.5**2)*noise_model_channel(rho, error_type, error)

        else:
            circuit = cirq.Circuit(gate).unitary()
            rho = (.5**2)*(circuit @ rho) @ (circuit.conj().T)

        rho_k.append(rho)

    for j_tup in it.product(range(len(Q_ops)), repeat=2):
        for k in range(len(rho_k)):
            Q_j = np.kron(Q_ops[j_tup[0]]._unitary_(),
                          Q_ops[j_tup[1]]._unitary_())

            Q_rho = Q_j@rho_k[k]

            # if noisy:
            #    Q_rho = noise_model_channel(Q_rho, error_type, error)

            Ojk = np.trace(Q_rho)
            O_q[list(it.product(range(len(Q_ops)), repeat=2)
                     ).index(j_tup)][k] = Ojk

    return O_q


def lin_comb(op, basis):
    '''
    Generates quasi-probability distribution from an operator and operator basis.

    Input:
    op         [array] : (4^n, 4^n) operator
    basis [list/array] : (1,16^n) list/array of (4^n, 4^n) basis operators for decomposition

    Output:
    x [array] : array of quasi-probability coefficients

    '''
    y = [b for b in basis]
    basis_vec = np.array([b.flatten('F') for b in basis])

    A = np.array([b_i for b_i in basis_vec]).T
    b = op.flatten('F')

    x = np.linalg.solve(A, b)

    return x


class QPSamp():

    def __init__(self, circuit, error_type, error):

        self.B_ops_matrix = basis_ops()
        self.B_ops_circuit = basis_ops_sim()

        # Assume perfect knowledge of basis ops PTM for now...
        _dummy_qubit = list(circuit.all_qubits())[0]
        applied_list = [apply_to_qubit(op, _dummy_qubit) for op in self.B_ops_matrix]
        self.B_bar_1q = [o_tilde_1q(app_on, noisy=False, error_type=error_type, error=0).real for app_on in applied_list]
        # self.B_bar_1q = [o_tilde_1q(apply_to_qubit(
        #     op, _dummy_qubit), noisy=False, error_type=error_type, error=0).real for op in self.B_ops_matrix]
        self.B_bar_2q = [np.kron(b1, b2)
                         for b1, b2 in it.product(self.B_bar_1q, repeat=2)]

        self.cost = []
        self.qp_dists = []

        self.error_type = error_type
        self.error = error

        self.circuit = circuit
        self.circuit_noisy = noisy_circuit(
            circuit, error_type=error_type, error=error)

    def B_ops_circuit_2q(self, qubits, index):
        return [apply_to_qubit(op[0], qubits[0]) + apply_to_qubit(op[1], qubits[1]) for op in it.product(self.B_ops_circuit, repeat=2)][index]

    def qp_costs(self):

        # Doesn't matter which qubit to choose, we're only interested in the unitary matrix.
        qubit = list(self.circuit.all_qubits())[0]

        # Generate QP distributions for every operator appearing in circuit
        for op in self.circuit.all_operations():
            if len(op.qubits) == 1:
                o = o_tilde_1q(op, noisy=False,
                               error_type=self.error_type, error=0)
                o_noise = o_tilde_1q(
                    op, noisy=True, error_type=self.error_type, error=self.error)

                N_inv = o.dot(np.linalg.inv(o_noise))

                B_ops = self.B_bar_1q

            elif len(op.qubits) == 2:
                o = o_tilde_2q(op, noisy=False,
                               error_type=self.error_type, error=0)
                o_noise = o_tilde_2q(
                    op, noisy=True, error_type=self.error_type, error=self.error)

                N_inv = o.dot(np.linalg.inv(o_noise))

                # Replace with B_ops for relevant qubits when reconstructing using GST
                # For this case it doesn't matter
                B_ops = self.B_bar_2q

            # Generate QP distribution for the operator
            self.qp_dists.append(lin_comb(N_inv, B_ops))

            if type(op.gate).__name__ != 'MeasurementGate':
                self.cost.append(sum(np.abs(lin_comb(N_inv, B_ops))))
            else:
                self.cost.append(1)

    def sampled_measurement(self, meas_reps, sample_reps):

        exp_sampled_trial = []

        # Sample circuits.
        # Will be parallelized once bug with operators is fixed.
        for rep in range(sample_reps):

            circuit_sampled = cirq.Circuit()
            qp_chosen = []

            for op in self.circuit.all_operations():

                circuit_sampled.append(op)

                if type(op.gate).__name__ != 'MeasurementGate':

                    circuit_sampled.append(noise_model(
                        dim=len(op.qubits), error_type=self.error_type, error=self.error)(*op.qubits))

                    # Replace with B_ops for relevant qubits when reconstructing using GST
                    # For this case it doesn't matter since we're not performing any tomography.
                    if len(op.qubits) == 1:
                        B_ops = self.B_bar_1q

                    elif len(op.qubits) == 2:
                        B_ops = self.B_bar_2q

                    # Load QP distribution for the operator
                    qp = self.qp_dists[list(
                        self.circuit.all_operations()).index(op)]

                    # Sample P distribution
                    prob_q = np.abs(
                        qp)/self.cost[list(self.circuit.all_operations()).index(op)]
                    B_ind = np.random.choice(range(len(B_ops)), p=prob_q)

                    qp_chosen.append(qp[B_ind])

                    if len(op.qubits) == 1:
                        circuit_sampled.append(apply_to_qubit(
                            self.B_ops_circuit[B_ind], op.qubits[0]))

                    elif len(op.qubits) == 2:
                        circuit_sampled.append(
                            self.B_ops_circuit_2q(op.qubits, B_ind))

                    if B_ind != 0:
                        circuit_sampled.append(noise_model(
                            dim=len(op.qubits), error_type=self.error_type, error=self.error)(*op.qubits))

            result_sampled = cirq.DensityMatrixSimulator().run(
                circuit_sampled, repetitions=meas_reps)
            exp_val = 2*result_sampled.histogram(key='M')[0]/meas_reps - 1
            exp_sampled_trial.append(np.sign(np.prod(qp_chosen))*exp_val)  # np.sign(np.prod(qp_chosen))*

        self.exp_sampled.append(np.mean(exp_sampled_trial))

    def ideal_measurement(self, meas_reps):

        result_nonoise = cirq.DensityMatrixSimulator().run(
            self.circuit, repetitions=meas_reps)

        self.exp_ideal.append(
            2*result_nonoise.histogram(key='M')[0]/meas_reps - 1)

    def noisy_measurement(self, meas_reps):

        result_noise = cirq.DensityMatrixSimulator().run(
            self.circuit_noisy, repetitions=meas_reps)

        self.exp_noisy.append(
            2*result_noise.histogram(key='M')[0]/meas_reps - 1)

    def run_experiment(self, meas_reps, sample_reps, stat_reps):

        self.qp_costs()

        for s_rep in tqdm(range(stat_reps)):

            self.exp_ideal = []
            self.exp_noisy = []
            self.exp_sampled = []

            self.ideal_measurement(meas_reps)
            self.noisy_measurement(meas_reps)

            self.sampled_measurement(meas_reps, sample_reps)

            print(len(self.cost))

            print('Ideal mean = {}'.format(np.mean(self.exp_ideal)))
            print('Noisy error = {}'.format(
                abs(np.mean(self.exp_ideal)-np.mean(self.exp_noisy))))
            temp = np.prod(self.cost)*np.mean(self.exp_sampled)
            print('Sampled error = {}'.format(
                abs(np.mean(self.exp_ideal)-np.prod(self.cost)*np.mean(self.exp_sampled))))

            # np.save('data/test_data/exp_swap_test.npy', self.exp_ideal)
            # np.save('data/test_data/exp_noisy_test.npy', self.exp_noisy)
            # np.save('data/test_data/exp_sampled_test.npy',
            #         [self.cost, self.exp_sampled])

        print(self.exp_sampled)

        import matplotlib.pyplot as plt
        # Plotting for the lol
        n_bins = 100
        fig, ax = plt.subplots(1, 1, tight_layout=True)
        ax.title.set_text('Reference code')
        ax.set_xlabel(f'measurement outcome (avg: {np.prod(self.cost)*np.mean(self.exp_sampled)})')  # X axis label
        ax.set_xlim([-1, 1])  # X axis limit
        ax.set_ylabel(f'Bin count')  # Y axis label
        # We can set the number of bins with the `bins` kwarg
        ax.hist(self.exp_sampled, bins=n_bins)

        plt.show()
