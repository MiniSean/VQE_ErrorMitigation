# Random parameter sampling from probability distribution
import cirq
import numpy as np
from typing import Tuple, Union, Callable, Iterator, List
from scipy.sparse import csc_matrix, lil_matrix, linalg, vstack

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

    # Get unitary generator
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

    # Get GST transformed unitary
    @staticmethod
    def get_gst_basis_set() -> Iterator[csc_matrix]:
        """Yields all gst transformed basis unitaries"""
        for unitary in BasisUnitarySet.get_basis_unitary_set():
            yield gate_set_tomography(unitary)[1]

    # Get operator generator
    @staticmethod
    def get_basis_operator_set() -> Iterator[cirq.SingleQubitGate]:
        """Yields all basis operators in order."""
        for unitary in BasisUnitarySet.get_basis_unitary_set():
            yield IBasisOperator(unitary=unitary)

    @staticmethod
    def get_quasi_probabilities(gate: Union[np.ndarray, cirq.Gate]) -> Tuple[List[float], List[np.ndarray]]:
        """
        Computes gate set tomography of provided gate/unitary.
        :return: quasi-probabilities based on this and BasisUnitarySet basis set.
        """
        if isinstance(gate, cirq.Gate):  # Works with gate unitary
            gate = cirq.unitary(gate)
        gate = gate_set_tomography(gate)[1].toarray()  # Apply GST
        basis = [basis.toarray() for basis in BasisUnitarySet.get_gst_basis_set()]  # Convert sparse basis to array
        return get_quasiprobabilities(target=gate, basis_set=basis), basis


class IBasisOperator(cirq.SingleQubitGate):

    def __init__(self, unitary: np.ndarray, exponent=1):
        self._unitary = unitary
        self._exponent = exponent

    def _unitary_(self):
        return np.linalg.matrix_power(self._unitary, self._exponent)


def gate_set_tomography(gate: Union[np.ndarray, cirq.Gate]) -> Tuple[csc_matrix, csc_matrix]:
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
    if isinstance(gate, cirq.Gate):  # Works with gate unitary
        gate = cirq.unitary(gate)
    d = 2  # Single qubit gates (d=4 in case of Two qubit gates)
    # Initial state vectors (with normalization) and observable matrices
    t_map = csc_matrix([[1, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1], [1, -1, 0, 0]])  # Mapping transform
    observable_set = [csc_matrix([[1, 0], [0, 1]]), csc_matrix([[0, 1], [1, 0]]), csc_matrix([[0, -1j], [1j, 0]]), csc_matrix([[1, 0], [0, -1]])]
    initial_density_set = [(1/d) * (gate @ (observable @ gate.conj().transpose())) for observable in observable_set]  # Pauli Transfer Matrix

    # Initiate output matrices
    o_tilde = lil_matrix((len(initial_density_set), len(observable_set)), dtype=complex)  # Bias operator O
    g = lil_matrix((len(initial_density_set), len(observable_set)), dtype=complex)  # Non-bias operator g
    for j, obs in enumerate(observable_set):
        for k, rho in enumerate(initial_density_set):
            o_tilde[j, k] = (obs @ (gate @ rho)).diagonal().sum()  # Trace
            g[j, k] = (obs @ rho).diagonal().sum()  # Trace

    # Map lil matrix to csc matrix
    o_tilde = o_tilde.tocsc()
    g = g.tocsc()

    # Define estimator
    # try:
    #     g_inv = linalg.inv(g)
    #     t_inv = linalg.inv(t_map)
    #     gate_estimator = t_map @ linalg.inv(g) @ o_tilde @ linalg.inv(t_map)  # Estimator gate O
    # except Exception:
    #     print(f'Unable to create estimator.\n{gate}\n')

    return o_tilde, g  # gate_estimator


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
    pure_gate = cirq.H

    start_time = time.time()
    basis_set = list(BasisUnitarySet.get_basis_unitary_set())
    gst_target = gate_set_tomography(pure_gate)[1].toarray()
    qp, basis = BasisUnitarySet.get_quasi_probabilities(gate=pure_gate)
    rcs_target = reconstruct_from_basis(qp, basis)
    print(f'Calculation time: {time.time() - start_time} sec.')
    print(f'Sum of quasi-probabilities: {sum(qp)}')
    print(f'Target vs Reconst. difference: {get_matrix_difference(rcs_target, gst_target)}')

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
