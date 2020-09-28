# Random parameter sampling from probability distribution
import cirq
import numpy as np
from typing import Tuple, Union, Callable, Iterator, List

MEAN = 0.0
STD = 1.0


# Contains all 16 basis unitary operations
class BasisUnitarySet:
    """
    Single qubit operations.
    Basis unitaries are constructed based on Piz, Rx and Rz.
    Which are constructed based on H and S-phase.
    :return: Generator over all (single qubit) basis unitary
    """
    # Base Constructors
    # Hadamard
    @staticmethod
    def _h_unitary() -> np.ndarray:
        return cirq.unitary(cirq.H)

    # S phase
    @staticmethod
    def _s_unitary() -> np.ndarray:
        return (1/np.sqrt(2))*(cirq.unitary(cirq.I) - 1j*cirq.unitary(cirq.Z))

    # Pi z
    @staticmethod
    def piz_unitary() -> np.ndarray:
        return .5*(cirq.unitary(cirq.I) + cirq.unitary(cirq.Z))

    # R z
    @staticmethod
    def rz_unitary() -> np.ndarray:
        return np.linalg.matrix_power(BasisUnitarySet._s_unitary(), 3)

    # R x
    @staticmethod
    def rx_unitary() -> np.ndarray:
        return BasisUnitarySet._h_unitary() @ (BasisUnitarySet.rz_unitary() @ BasisUnitarySet._h_unitary())

    # Additional unitary set
    # Identity
    @staticmethod
    def i_unitary() -> np.ndarray:
        return cirq.unitary(cirq.I)

    # Sigma x
    @staticmethod
    def sx_unitary() -> np.ndarray:
        return np.linalg.matrix_power(BasisUnitarySet.rx_unitary(), 2)

    # Sigma y
    @staticmethod
    def sy_unitary() -> np.ndarray:
        return BasisUnitarySet.sx_unitary() @ BasisUnitarySet.sz_unitary()

    # Sigma z
    @staticmethod
    def sz_unitary() -> np.ndarray:
        return np.linalg.matrix_power(BasisUnitarySet.rz_unitary(), 2)

    # R y
    @staticmethod
    def ry_unitary() -> np.ndarray:
        return np.linalg.matrix_power(BasisUnitarySet.rz_unitary(), 3) @ (BasisUnitarySet.rx_unitary() @ BasisUnitarySet.rz_unitary())

    # R y z
    @staticmethod
    def ryz_unitary() -> np.ndarray:
        return BasisUnitarySet.rx_unitary() @ np.linalg.matrix_power(BasisUnitarySet.rz_unitary(), 2)

    # R z x
    @staticmethod
    def rzx_unitary() -> np.ndarray:
        return BasisUnitarySet.rz_unitary() @ (BasisUnitarySet.rx_unitary() @ BasisUnitarySet.rz_unitary())

    # R x y
    @staticmethod
    def rxy_unitary() -> np.ndarray:
        return np.linalg.matrix_power(BasisUnitarySet.rx_unitary(), 2) @ BasisUnitarySet.rz_unitary()

    # Pi x
    @staticmethod
    def pix_unitary() -> np.ndarray:
        return np.linalg.matrix_power(BasisUnitarySet.rz_unitary(), 3) @ (np.linalg.matrix_power(BasisUnitarySet.rx_unitary(), 3) @ (BasisUnitarySet.piz_unitary() @ (BasisUnitarySet.rx_unitary() @  BasisUnitarySet.rz_unitary())))

    # Pi y
    @staticmethod
    def piy_unitary() -> np.ndarray:
        return BasisUnitarySet.rx_unitary() @ (BasisUnitarySet.piz_unitary() @ np.linalg.matrix_power(BasisUnitarySet.rx_unitary(), 3))

    # Pi y z
    @staticmethod
    def piyz_unitary() -> np.ndarray:
        return np.linalg.matrix_power(BasisUnitarySet.rz_unitary(), 3) @ (np.linalg.matrix_power(BasisUnitarySet.rx_unitary(), 3) @ (BasisUnitarySet.piz_unitary() @ (np.linalg.matrix_power(BasisUnitarySet.rx_unitary(), 3) @  BasisUnitarySet.rz_unitary())))

    # Pi z x
    @staticmethod
    def pizx_unitary() -> np.ndarray:
        return BasisUnitarySet.piy_unitary() @ np.linalg.matrix_power(BasisUnitarySet.rz_unitary(), 2)

    # Pi x y
    @staticmethod
    def pixy_unitary() -> np.ndarray:
        return BasisUnitarySet.piz_unitary() @ np.linalg.matrix_power(BasisUnitarySet.rx_unitary(), 2)

    # Get unitary generator for single qubit gates
    @staticmethod
    def get_basis_unitary_set() -> Iterator[np.ndarray]:
        """Yields all basis unitary operations in order."""
        yield BasisUnitarySet.i_unitary()
        yield BasisUnitarySet.sx_unitary()
        yield BasisUnitarySet.sy_unitary()
        yield BasisUnitarySet.sz_unitary()
        yield BasisUnitarySet.rx_unitary()
        yield BasisUnitarySet.ry_unitary()
        yield BasisUnitarySet.rz_unitary()
        yield BasisUnitarySet.ryz_unitary()
        yield BasisUnitarySet.rzx_unitary()
        yield BasisUnitarySet.rxy_unitary()
        yield BasisUnitarySet.pix_unitary()
        yield BasisUnitarySet.piy_unitary()
        yield BasisUnitarySet.piz_unitary()
        yield BasisUnitarySet.piyz_unitary()
        yield BasisUnitarySet.pizx_unitary()
        yield BasisUnitarySet.pixy_unitary()

    # Get unitary generator for two qubit gates
    @staticmethod
    def get_basis_tensor_unitary_set() -> Iterator[np.ndarray]:
        """Yields all tensor products of each basis unitary operations in order."""
        for unitary_A in BasisUnitarySet.get_basis_unitary_set():
            for unitary_B in BasisUnitarySet.get_basis_unitary_set():
                yield np.kron(unitary_A, unitary_B)  # Tensor product in correct format

    # Get basis observable set for GST (single/double qubit gates)
    @staticmethod
    def get_observable_set(dim: int) -> Iterator[np.ndarray]:
        """Yields all basis observables in order depending on gate dimension"""
        def _basis() -> Iterator[np.ndarray]:
            yield np.array([[1, 0], [0, 1]])  # I
            yield np.array([[0, 1], [1, 0]])  # X
            yield np.array([[0, -1j], [1j, 0]])  # Y
            yield np.array([[1, 0], [0, -1]])  # Z

        if dim == 2:  # (single qubit)
            for observable in _basis():
                yield observable
        elif dim == 4:  # (double qubits)
            for observable_A in _basis():
                for observable_B in _basis():
                    yield np.kron(observable_A, observable_B)  # Tensor product in correct format
        else:
            raise NotImplemented

    # Get initial state set for GST (single/double qubit gates)
    @staticmethod
    def get_initial_state_set(gate: np.ndarray) -> Iterator[np.ndarray]:
        """Yields all initial states in order depending on gate shape"""
        if gate.shape[0] == gate.shape[1] == 2:  # Pauli Transfer Matrix (single qubit)
            for observable in BasisUnitarySet.get_observable_set(dim=2):
                yield .5 * (gate @ (observable @ gate.conj().transpose()))
        elif gate.shape[0] == gate.shape[1] == 4:  # Pauli Transfer Matrix (double qubits)
            for observable in BasisUnitarySet.get_observable_set(dim=4):
                yield .25 * (gate @ (observable @ gate.conj().transpose()))
        else:
            raise NotImplemented

    # Get GST transformed unitary
    @staticmethod
    def get_gst_basis_set(dim: int) -> Iterator[np.ndarray]:
        """Yields all gst transformed basis unitaries depending on gate dimension"""
        if dim == 2:  # (single qubit)
            for unitary in BasisUnitarySet.get_basis_unitary_set():
                yield BasisUnitarySet.gate_set_tomography(unitary)[1]
        elif dim == 4:  # (double qubits)
            for unitary in BasisUnitarySet.get_basis_tensor_unitary_set():
                yield BasisUnitarySet.gate_set_tomography(unitary)[1]
        else:
            raise NotImplemented

    # Get operator generator
    @staticmethod
    def get_basis_operator_set() -> Iterator[cirq.SingleQubitGate]:
        """Yields all basis operators in order."""
        for unitary in BasisUnitarySet.get_basis_unitary_set():
            yield IBasisOperator(unitary=unitary)

    @staticmethod
    def gate_set_tomography(gate: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Implements single qubit gate GST.
        Using mapping matrix T:
        [[1, 1, 1, 1],
         [0, 0, 1, 0],
         [0, 0, 0, 1],
         [1, -1, 0, 0]]
        :type gate: Gate (with noise) to apply GST on
        :return: Õ, g, Ô (estimator)
        """
        output_shape = (gate.shape[0]**2, gate.shape[1]**2)  # Temp

        # Initial state vectors (with normalization) and observable matrices
        # t_map = csc_matrix([[1, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1], [1, -1, 0, 0]])  # Mapping transform
        _observable_set = list(BasisUnitarySet.get_observable_set(dim=gate.shape[0]))
        _initial_state_set = list(BasisUnitarySet.get_initial_state_set(gate=gate))

        # Initiate output matrices
        o_tilde = np.zeros(output_shape, dtype=complex)  # Bias operator O
        g = np.zeros(output_shape, dtype=complex)  # Non-bias operator g
        for j, obs in enumerate(_observable_set):
            for k, rho in enumerate(_initial_state_set):
                o_tilde[j, k] = np.trace(obs @ (gate @ rho))  #.diagonal().sum()  # Trace
                g[j, k] = np.trace(obs @ rho)  #.diagonal().sum()  # Trace

        # Map lil matrix to csc matrix
        # o_tilde = o_tilde.tocsc()
        # g = g.tocsc()

        # Define estimator
        # try:
        #     g_inv = linalg.inv(g)
        #     t_inv = linalg.inv(t_map)
        #     gate_estimator = t_map @ linalg.inv(g) @ o_tilde @ linalg.inv(t_map)  # Estimator gate O
        # except Exception:
        #     print(f'Unable to create estimator.\n{gate}\n')

        return o_tilde, g  # gate_estimator

    @staticmethod
    def get_quasiprobabilities(target: np.ndarray, basis_set: List[np.ndarray]) -> List[float]:
        if target.shape != basis_set[0].shape:
            raise ValueError(f'Target gate \n{target}\n and Basis set operation matrices should have the same shape.')
        basis_set_matrix = np.array([basis.flatten('F') for basis in basis_set]).transpose()
        target_vector = target.flatten('F')
        # Catch exception
        try:
            return [v.real for v in np.linalg.solve(basis_set_matrix, target_vector)]  # Solve system of linear equations
        except np.linalg.LinAlgError:
            print(f'WARNING: Not able to solve system of linear equations with ill defined basis.')
            return [0. for i in basis_set]

    # Super function combining several functionality's
    @staticmethod
    def get_decomposition(gate: Union[np.ndarray, cirq.Gate]) -> Tuple[List[float], List[np.ndarray]]:
        """
        Determines single or double qubit gate.
        Computes gate set tomography of provided gate/unitary.
        :return: quasi-probabilities based on this and BasisUnitarySet basis set.
        """
        if isinstance(gate, cirq.Gate):  # Works with gate unitary
            gate = cirq.unitary(gate)

        basis = list(BasisUnitarySet.get_gst_basis_set(dim=gate.shape[0]))  # Convert sparse basis to array
        gate = BasisUnitarySet.gate_set_tomography(gate)[1]  # Apply GST
        return BasisUnitarySet.get_quasiprobabilities(target=gate, basis_set=basis), basis


class IBasisOperator(cirq.SingleQubitGate):

    def __init__(self, unitary: np.ndarray, exponent=1):
        self._unitary = unitary
        self._exponent = exponent

    def _unitary_(self):
        return np.linalg.matrix_power(self._unitary, self._exponent)


def reconstruct_from_basis(quasi_probabilities: List[float], basis_set: List[np.ndarray]) -> np.ndarray:
    if len(quasi_probabilities) != len(basis_set):
        raise ValueError(f'Each basis should have a corresponding quasi-probability value')
    return sum(i[0] * i[1] for i in zip(quasi_probabilities, basis_set))  # Dot product of lists


def get_matrix_difference(A: np.ndarray, B: np.ndarray) -> float:
    """
    Checks if both matrices have the same shape.
    Calculates the difference matrix, sums over elements, takes modules and divides by number of elements.
    :param A: 2D Matrix
    :param B: 2D Matrix
    :return: Difference parameter
    """
    if A.shape != B.shape:
        raise TypeError('Density matrices do not have same dimensions.')
    D = A - B
    d = np.sum(D)
    d = np.sqrt(d.real**2 + d.imag**2)  # Modules
    return d / (D.shape[0] * D.shape[1])  # Normalize in matrix length


# Probability distributions
def get_probability_normal(var_dim: Union[int, Tuple[int]]) -> Union[float, np.ndarray]:
    """
    Characterized by type identity: Callable[[Union[int, Tuple[int]]], Union[float, np.ndarray]]
    :param var_dim: Dimension of output probabilities
    :return: Outputs set of random (normal distributed) probability depending dimensions specified
    """
    return np.random.normal(loc=MEAN, scale=STD, size=var_dim)


def get_probability_uniform(var_dim: Union[int, Tuple[int]]) -> Union[float, np.ndarray]:
    """
    Characterized by type identity: Callable[[Union[int, Tuple[int]]], Union[float, np.ndarray]]
    :param var_dim: Dimension of output probabilities
    :return: Outputs set of random (uniform distributed) probability depending dimensions specified
    """
    return np.random.random_sample(size=var_dim)  # np.random.uniform(low=0.0, high=1.0, size=var_dim)


# Monte Carlo sampling precision
def get_samples(precision: float, gamma: float, distribution: Callable[[Union[int, Tuple[int]]], Union[float, np.ndarray]]) -> Union[float, np.ndarray]:
    if precision == 0:
        raise ZeroDivisionError
    sample_count = int(np.sqrt(gamma / precision))  # Monte Carlo sample count
    return distribution(sample_count)


if __name__ == '__main__':
    import time
    # from QP_QEM_lib import o_tilde_1q, basis_ops, apply_to_qubit
    dummy_qubits = cirq.LineQubit.range(2)
    two_qubit_gate = cirq.CNOT(dummy_qubits[0], dummy_qubits[1])
    pure_gate = cirq.unitary(cirq.H)

    cnot_unitary = cirq.unitary(two_qubit_gate)

    gst_cnot = BasisUnitarySet.gate_set_tomography(gate=cnot_unitary)[1]  # For reference
    print(gst_cnot)

    # --------------------------

    start_time = time.time()
    qp, basis = BasisUnitarySet.get_decomposition(gate=cnot_unitary)
    rcs_target = reconstruct_from_basis(qp, basis)
    print(f'Calculation time: {time.time() - start_time} sec.')
    print(f'Sum of quasi-probabilities: {sum(qp)}')
    gst_target = BasisUnitarySet.gate_set_tomography(gate=cnot_unitary)[1]  # For reference
    print(f'Target vs Reconst. difference: {get_matrix_difference(rcs_target, gst_target)}')

    # --------------------------

    #
    # # --------------------------
    #
    # # Reference GST transformation on basis sets
    # # Requires an applied gate
    # dummy_qubit = cirq.LineQubit(0)
    # error_type = 'depolarize'
    #
    # operator = IBasisOperator(cirq.unitary(pure_gate))
    # operator = operator.on(dummy_qubit)
    # gst_target_ref = o_tilde_1q(gate=operator, noisy=False, error_type=error_type, error=0)
    #
    # applied_list = [apply_to_qubit(op, dummy_qubit) for op in basis_ops()]
    # gst_basis_ref = [o_tilde_1q(app_on, noisy=False, error_type=error_type, error=0).real for app_on in applied_list]
    #
    # print('Test basis set equality:')
    # basis_set_ref = [cirq.Circuit(gate).unitary() for gate in applied_list]
    # print(np.array(basis_set) == np.array(basis_set_ref))
    # # --------------------------
    #
    # # Local GST transformation on basis sets
    # gst_target = gate_set_tomography(pure_gate)[1].toarray()
    # gst_basis = [gate_set_tomography(u)[1].toarray() for u in BasisUnitarySet.get_basis_unitary_set()]
    #
    # print('Test GST target equality:')
    # print(gst_target == gst_target_ref)
    # print('Test GST basis set equality:')
    # print(np.array(gst_basis) == np.array(gst_basis_ref))
    #
    # # --------------------------
    #
    # print('target:')
    # print(gst_target)
    # quasi_probabilities = get_quasiprobabilities(gst_target, gst_basis)
    # print('probabilities:')
    # print(quasi_probabilities)
    # result = reconstruct_from_basis(quasi_probabilities, gst_basis)
    # print('reconstruction:')
    # print(result)
    # print('Test gate reconstruction from quasi probabilities')
    # print(np.array(gst_target) == np.array(result))
    #
    # # --------------------------
