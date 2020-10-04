# Random parameter sampling from probability distribution
import random
import cirq
import numpy as np
from typing import Tuple, Union, Callable, Iterator, List, Dict
from src.data_containers.helper_interfaces.i_noise_wrapper import INoiseModel

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

    # Depricated
    # Get unitary generator for two qubit gates
    # @staticmethod
    # def get_basis_tensor_unitary_set() -> Iterator[np.ndarray]:
    #     """Yields all tensor products of each basis unitary operations in order."""
    #     for unitary_A in BasisUnitarySet.get_basis_unitary_set():
    #         for unitary_B in BasisUnitarySet.get_basis_unitary_set():
    #             yield np.kron(unitary_A, unitary_B)  # Tensor product in correct format

    @staticmethod
    def get_t_map(dim: int) -> np.ndarray:
        t_map = np.array([[1, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1], [1, -1, 0, 0]])  # Mapping transform
        if dim == 2:  # (single qubit)
            return t_map
        elif dim == 4:  # (double qubits)
            return np.kron(t_map, t_map)
        else:
            raise NotImplemented

    # Get basis observable set for GST (single/double qubit gates)
    @staticmethod
    def get_observable_set(dim: int) -> Iterator[np.ndarray]:
        """Yields all basis observables in order depending on gate dimension"""
        def _single_qubit() -> Iterator[np.ndarray]:
            yield np.array([[1, 0], [0, 1]])  # I
            yield np.array([[0, 1], [1, 0]])  # X
            yield np.array([[0, -1j], [1j, 0]])  # Y
            yield np.array([[1, 0], [0, -1]])  # Z

        if dim == 2:  # (single qubit)
            for observable in _single_qubit():
                yield observable
        elif dim == 4:  # (double qubits)
            for observable_A in _single_qubit():
                for observable_B in _single_qubit():
                    yield np.kron(observable_A, observable_B)  # Tensor product in correct format
        else:
            raise NotImplemented

    # Get initial state set for GST (single/double qubit gates)
    @staticmethod
    def get_initial_state_set(gate: np.ndarray) -> Iterator[np.ndarray]:
        """Yields all initial states in order depending on gate shape"""
        # First tensor product then pauli transfer matrix
        if gate.shape[0] == gate.shape[1] == 2:  # Pauli Transfer Matrix (single qubit)
            for observable in BasisUnitarySet.get_observable_set(dim=2):
                yield .5 * (gate @ (observable @ gate.conj().transpose()))
        elif gate.shape[0] == gate.shape[1] == 4:  # Pauli Transfer Matrix (double qubits)
            for observable in BasisUnitarySet.get_observable_set(dim=4):
                yield .25 * (gate @ (observable @ gate.conj().transpose()))
        else:
            raise NotImplemented

    # Get normal unitary set
    @staticmethod
    def get_basis_set(dim: int) -> Iterator[np.ndarray]:
        """Yields all gst transformed basis unitaries depending on gate dimension"""
        def _single_qubit() -> Iterator[np.ndarray]:
            for unitary in BasisUnitarySet.get_basis_unitary_set():
                yield unitary

        # First gate set tomography then tensor product
        if dim == 2:  # (single qubit)
            for basis in _single_qubit():
                yield basis
        elif dim == 4:  # (double qubits)
            for basis_A in _single_qubit():
                for basis_B in _single_qubit():
                    yield np.kron(basis_A, basis_B)  # Tensor product in correct format
        else:
            raise NotImplemented

    # Get GST transformed unitary
    @staticmethod
    def get_gst_basis_set(dim: int) -> Iterator[np.ndarray]:
        """Yields all gst transformed basis unitaries depending on gate dimension"""
        def _single_qubit() -> Iterator[np.ndarray]:
            for unitary in BasisUnitarySet.get_basis_unitary_set():
                yield BasisUnitarySet.gate_set_tomography(unitary)

        # First gate set tomography then tensor product
        if dim == 2:  # (single qubit)
            for basis in _single_qubit():
                yield basis
        elif dim == 4:  # (double qubits)
            for basis_A in _single_qubit():
                for basis_B in _single_qubit():
                    yield np.kron(basis_A, basis_B)  # Tensor product in correct format
        else:
            raise NotImplemented

    # Get operator generator
    @staticmethod
    def get_basis_operator_set() -> Iterator[cirq.SingleQubitGate]:
        """Yields all basis operators in order."""
        for unitary in BasisUnitarySet.get_basis_unitary_set():
            yield IBasisGateSingle(unitary=unitary)

    # Apply subsequent noise model to gate before applying this function
    @staticmethod
    def gate_set_tomography(gate: np.ndarray, noise_wrapper: Callable[[np.ndarray], np.ndarray] = None) -> np.ndarray:  # Tuple[np.ndarray, np.ndarray]:
        """
        Implements single and double qubit gate GST.
        Using mapping matrix T:
        [[1, 1, 1, 1],
         [0, 0, 1, 0],
         [0, 0, 0, 1],
         [1, -1, 0, 0]]
        :type gate: Gate (without noise) to apply GST on
        :param noise_wrapper: Noise wrapper for Initial state matrices
        :return: Ô (Estimator)
        """

        # Initial state vectors (with normalization) and observable matrices
        # Q_j = I, X, Y, Z (Pauli matrices)
        # Rho_i = Q_j
        # Rho_PTM_i = 1/d gate Q_j gate^\dagger (Pauli transfer matrix)
        _t_map = BasisUnitarySet.get_t_map(dim=gate.shape[0])  # Mapping transform
        _observable_set = list(BasisUnitarySet.get_observable_set(dim=gate.shape[0]))  # 4^n linearly independent observables
        _initial_state_set_ptm = list(BasisUnitarySet.get_initial_state_set(gate=gate))  # 4^n linearly independent initial states
        if noise_wrapper is not None:  # Apply noise to rho basis set
            _initial_state_set_ptm = [noise_wrapper(rho) for rho in _initial_state_set_ptm]

        # Initiate output matrices
        # Õ = M_out gate M_in
        # M_out = <<Q_j|Pauli_i>>
        # M_in = <<Pauli_j|Rho_i>>
        # g = M_out M_in
        # Ô = T g^-1 Õ T^-1
        # Ô_PTM = <<Q_j|Rho_PTM_i>>
        output_shape = (gate.shape[0]**2, gate.shape[1]**2)
        ptm_estimation = np.zeros(output_shape, dtype=complex)  # Estimation matrix Ô, build with Pauli Transfer Matrix transform on Rho_i basis set
        # expectation = np.zeros(output_shape, dtype=complex)  # Expectation matrix Õ, equal to 'gate' up to transformation
        # basis_map_g = np.zeros(output_shape, dtype=complex)  # Non-bias mapping matrix g
        for j, obs in enumerate(_observable_set):
            for k, rho in enumerate(_initial_state_set_ptm):
                ptm_estimation[j, k] = np.trace(obs @ rho)  # Trace
                # expectation[j, k] = np.trace(obs @ (gate @ rho))  # Trace
                # basis_map_g[j, k] = np.trace(obs @ rho)  # Trace

        # print(f'g:')
        # print(basis_map_g)
        # print(f'Õ:')
        # print(expectation)
        # # Define estimator
        # try:
        #     g_inv = np.linalg.inv(basis_map_g)
        #     t_inv = np.linalg.inv(_t_map)
        #     gate_estimation = _t_map @ (g_inv @ (expectation @ t_inv))  # Estimator gate O
        #     print(f'Ô:')
        #     print(gate_estimation)
        #     print(f'Test g:')
        #     print(test_g)
        #     return gate_estimation  # expectation, basis_map_g
        # except np.linalg.LinAlgError:
        #     print(f'WARNING: Not able to invert g matrix.')
        #     return expectation
        return ptm_estimation

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
            exception_result = [0. for i in basis_set]  # Only identity matrix
            exception_result[0] = 1
            return exception_result


class IBasisGateSingle(cirq.SingleQubitGate):

    def __init__(self, unitary: np.ndarray, exponent=1):
        self._unitary = unitary
        self._exponent = exponent

    def _unitary_(self):
        return np.linalg.matrix_power(self._unitary, self._exponent)

    def __str__(self):
        return 'Su.Op.'  # Super operator


class IBasisGateTwo(cirq.TwoQubitGate):

    def __init__(self, unitary: np.ndarray, exponent=1):
        self._unitary = unitary
        self._exponent = exponent

    def _unitary_(self):
        return np.linalg.matrix_power(self._unitary, self._exponent)

    def __str__(self):
        return 'Su.Op.'  # Super operator


class QuasiCircuitIdentifier:

    def __init__(self, circuit: cirq.Circuit, gate_lookup: Dict[cirq.Gate, List[float]]):
        # Construct identifier indices
        self.identifiers, self.sign, self.C = QuasiCircuitIdentifier._get_identifiers(circuit=circuit, gate_lookup=gate_lookup)
        # print(self.identifiers)

    @staticmethod
    def _get_identifiers(circuit: cirq.Circuit, gate_lookup: Dict[cirq.Gate, List[float]]) -> Tuple[List[int], int, float]:
        id_list = list()  # Order specific identifier collection
        id_sign = 1
        id_weight = 1
        for operator in circuit.all_operations():
            if type(operator.gate).__name__ == 'MeasurementGate':
                continue
            qp = gate_lookup[operator.gate]
            id, p_sign, weight = QuasiCircuitIdentifier._get_identifier(quasi_probabilities=qp)
            id_list.append(id)  # Append basis id to result list
            id_sign *= p_sign  # Multiply probability sign to result sign
            id_weight *= weight  # Multiply quasi probability weight
        return id_list, id_sign, id_weight

    @staticmethod
    def _get_identifier(quasi_probabilities: List[float]) -> Tuple[int, int, float]:
        """Returns random index from quasi probability list depending on probability distribution, its sign and weight"""
        c_weight = sum([abs(p) for p in quasi_probabilities])
        p_indicator = random.uniform(0, 1) * c_weight  # Scales random from [0, c_weight]
        for i, probability in enumerate(quasi_probabilities):
            p_indicator -= abs(probability)
            if p_indicator <= 0:
                return i, np.sign(probability), c_weight
        last_index = len(quasi_probabilities) - 1
        return last_index, np.sign(quasi_probabilities[last_index]), c_weight

    def __eq__(self, other):
        return other.__hash__() == self.__hash__()

    def __hash__(self):
        return hash(tuple(self.identifiers))


class IErrorMitigationManager:
    """Manages family of circuits to be measured"""

    def get_mu_effective(self) -> Tuple[List[float], float]:
        """
        Constructs circuits based on identifiers, measures them once, multiplies by the times the identifier appears.
        Calculates effective average.
        :return: Effective average of circuit measurement values.
        """
        result = list()
        C_weight = 1
        simulator = cirq.DensityMatrixSimulator(ignore_measurement_results=False)  # Mixed state simulator
        for key, value in self._dict_identifiers.items():
            circuit_to_run = IErrorMitigationManager.get_updated_circuit(clean_circuit=self.circuit, noise_wrapper=self._noise_wrapper, circuit_identifier=key)
            # Setup simulator
            simulated_result = simulator.simulate(program=circuit_to_run)  # Include final density matrix
            # Get final density matrix trace
            trace = (np.array(simulated_result.final_density_matrix)).diagonal().sum()
            # Add for value amount
            for i in range(value):
                result.append(trace.real * key.sign)
                C_weight = key.C
        return result, np.mean(result) / C_weight

    # Temp
    def get_identifiers(self) -> Iterator[List[int]]:
        for id_class in self._dict_identifiers.keys():
            yield id_class.identifiers

    def __init__(self, clean_circuit: cirq.Circuit, noise_wrapper: Callable[[np.ndarray], np.ndarray]):
        # Store properties
        self.circuit = clean_circuit
        self._noise_wrapper = noise_wrapper
        self._gate_lookup = IErrorMitigationManager.get_qp_lookup(clean_circuit, noise_wrapper)
        self._dict_identifiers = {}

    # Magic function, adds range of 'QuasiCircuitIdentifier' to dictionary of identifiers
    def set_identifier_range(self, count: int):
        self._dict_identifiers = {}
        for i in range(count):
            self.add_circuit_identifier()

    # Magic function, adds 'QuasiCircuitIdentifier' to dictionary of identifiers
    def add_circuit_identifier(self):
        circuit_identifier = QuasiCircuitIdentifier(circuit=self.circuit, gate_lookup=self._gate_lookup)
        if circuit_identifier in self._dict_identifiers:
            self._dict_identifiers[circuit_identifier] += 1
        else:
            self._dict_identifiers[circuit_identifier] = 1

    @staticmethod
    def get_qp_lookup(clean_circuit: cirq.Circuit, noise_wrapper: Callable[[np.ndarray], np.ndarray]) -> Dict[cirq.Gate, List[float]]:
        """
        Iterates through the entire noiseless quantum circuit.
        Collect each gate unitary and performs a noise wrapper around the same unitary to get the noisy version.
        Both 'clean' and 'noisy' gates are mapped using gate set tomography.
        From this the inverse of the noise is calculated: N^-1 = O_clean @ O_noisy^-1.
        Both the quasi-probabilities as their weight is stored in a dictionary keyed by the gate itself.
        :param clean_circuit: Noiseless quantum circuit to iterate through
        :param noise_wrapper: Applies noise channel to single gate operation
        :return: Dictionary containing gate types as key and both the q-p list as their combined weight as values
        """
        result = {}  # gate to quasi probabilities
        for operator in clean_circuit.all_operations():
            if type(operator.gate).__name__ == 'MeasurementGate' or operator.gate in result.keys():
                continue
            # Get effective gates
            clean_gate = cirq.unitary(operator)  # Without noise
            # Get gate set tomography versions
            clean_gst_gate = BasisUnitarySet.gate_set_tomography(clean_gate)  # [1]
            noisy_gst_gate = BasisUnitarySet.gate_set_tomography(clean_gate, noise_wrapper=noise_wrapper)  # [1]
            # Get inverse noise
            N_inv = clean_gst_gate.dot(np.linalg.inv(noisy_gst_gate))

            basis_set = list(BasisUnitarySet.get_gst_basis_set(dim=clean_gate.shape[0]))  # Convert sparse basis_set to array
            qp = BasisUnitarySet.get_quasiprobabilities(target=N_inv, basis_set=basis_set)
            result[operator.gate] = qp
        return result

    @staticmethod
    def get_quasi_probability_induced_basis(basis: List[np.ndarray], quasi_probabilities: List[float]) -> np.ndarray:
        """
        Selects a random basis based on the corresponding quasi-probability.
        Random value is chosen from uniform distributed values [0, 1] and scaled using the qp summed weight.
        Classic pie-size selection scheme to retrieve the list specific index.
        Returns the corresponding basis.
        :param basis: List containing all basis matrices to choose from
        :param quasi_probabilities: List of singed probabilities to induce probability distribution
        :return: basis[i] where i is chosen based on the weighted probability distribution
        """
        if len(basis) != len(quasi_probabilities):
            raise IndexError('Each probability needs to represent a basis. Both list need to have the same length')
        c_weight = sum([abs(p) for p in quasi_probabilities])
        p_indicator = random.uniform(0, 1) * c_weight  # Scales random from [0, c_weight]
        print(f'Probability check landed on {p_indicator} in range [0, {c_weight})')  # Temporary print statements
        for i, probability in enumerate(quasi_probabilities):
            p_indicator -= abs(probability)
            if p_indicator <= 0:
                print(f'Using basis {i}')  # Temporary print statements
                return basis[i]
        return basis[-1]

    @staticmethod
    def get_updated_circuit(clean_circuit: cirq.Circuit, noise_wrapper: Callable[[np.ndarray], np.ndarray], circuit_identifier: QuasiCircuitIdentifier) -> cirq.Circuit:
        result = cirq.Circuit()
        for i, clean_operator in enumerate(clean_circuit.all_operations()):
            if type(clean_operator.gate).__name__ == 'MeasurementGate':
                result.append(clean_operator)  # Just in case
                continue
            unitary = cirq.unitary(clean_operator.gate)  # Get clean gate unitary
            # Use correct basis
            basis_dim = unitary.shape[0]
            basis_set = list(BasisUnitarySet.get_basis_set(basis_dim))
            # Build noisy and canceling basis
            noisy_basis = noise_wrapper(unitary)
            noise_canceling_basis = basis_set[circuit_identifier.identifiers[i]]
            # Construct operators
            qubit_list = clean_operator.qubits
            if len(qubit_list) == 1:
                noise_canceling_operator = IBasisGateSingle(unitary=noise_canceling_basis).on(*qubit_list)  # qubit_list[0]
                noisy_operator = IBasisGateSingle(unitary=noisy_basis).on(*qubit_list)  # qubit_list[0]
            elif len(qubit_list) == 2:
                noise_canceling_operator = IBasisGateTwo(unitary=noise_canceling_basis).on(*qubit_list)  # qubit_list[0], qubit_list[1]
                noisy_operator = IBasisGateTwo(unitary=noisy_basis).on(*qubit_list)
            else:
                raise NotImplemented('Can only construct single and two qubit operators')
            # Add to new circuit
            result.append([noisy_operator, noise_canceling_operator])
        return result

    def __str__(self):
        return self._dict_identifiers.__str__()


# Deprecated
def compose_gate_matrix(operator: cirq.Operation, noise_model: INoiseModel = None) -> np.ndarray:
    """
    Constructs matrix from gate and potential noise model.
    If no noise model is provided, simply call cirq.Gate._unitary_().
    Else, set up a density matrix simulator, run the gate + noise model and return the final density matrix.
    :param operator: cirq operation to be evaluated
    :param noise_model: custom noise model containing cirq noise operators.
    :return: (Unitary) matrix
    """
    qubit_count = len(operator.qubits)
    _dummy_qubits = cirq.LineQubit.range(qubit_count)
    simulator = cirq.DensityMatrixSimulator(ignore_measurement_results=True)  # Mixed state simulator
    _circuit = cirq.Circuit()
    # Apply operations and noise model to circuit
    if noise_model is None:
        _circuit.append(operator.gate.on(_dummy_qubits[0]))
        # return cirq.unitary(operator.gate)
    else:
        if qubit_count == 1:
            _circuit.append(operator.gate.on(_dummy_qubits[0]))
            _circuit.append([noise_gate.on(_dummy_qubits[0]) for noise_gate in noise_model.get_callable()()])
        elif qubit_count == 2:
            _circuit.append(operator.gate.on(_dummy_qubits[0], _dummy_qubits[1]))
            _circuit.append([noise_gate.on(_dummy_qubits[0]) for noise_gate in noise_model.get_callable()()])  # Only on first qubit?
            _circuit.append([noise_gate.on(_dummy_qubits[1]) for noise_gate in noise_model.get_callable()()])  # Or also on seccond?
        else:
            raise NotImplemented
    # Simulate circuit and return final density matrix
    simulated_result = simulator.simulate(program=_circuit)  # Include final density matrix
    return simulated_result.final_density_matrix


# Deprecated
# Super function combining several functionality's
def get_decomposition(gate: Union[np.ndarray, cirq.Gate]) -> Tuple[List[float], List[np.ndarray]]:
    """
    Determines single or double qubit gate.
    Computes gate set tomography of provided gate/unitary.
    :return: quasi-probabilities based on this and BasisUnitarySet basis set.
    """
    if isinstance(gate, cirq.Gate):  # Works with gate unitary
        gate = cirq.unitary(gate)

    basis = list(BasisUnitarySet.get_gst_basis_set(dim=gate.shape[0]))  # Convert sparse basis to array
    gate = BasisUnitarySet.gate_set_tomography(gate)  # Apply GST
    return BasisUnitarySet.get_quasiprobabilities(target=gate, basis_set=basis), basis


def reconstruct_from_basis(quasi_probabilities: List[float], basis_set: List[np.ndarray]) -> np.ndarray:
    """Reverse function of solving the linear set of equations"""
    if len(quasi_probabilities) != len(basis_set):
        raise ValueError(f'Each basis should have a corresponding quasi-probability value')
    return sum(i[0] * i[1] for i in zip(quasi_probabilities, basis_set))  # Dot product of lists


def get_hardcoded_noise(name: str, p: float) -> Callable[[np.ndarray], np.ndarray]:

    def gate_depolarize(gate: np.ndarray) -> np.ndarray:
        dim = gate.shape[0]
        result = np.zeros((dim, dim), dtype=complex)
        if dim == 2:
            result += (1 - (p * 4 / 3)) * gate  # I gate I^\dagger
            pauli_basis = BasisUnitarySet.get_observable_set(dim=dim)
            for basis in pauli_basis:
                result += (p / 3) * (basis @ (gate @ basis.conj().transpose()))  # P_i gate P_i^\dagger
            return result
        elif dim == 4:  # gate_2q_depolarize
            result += (1 - (p * 16 / 15)) * gate  # I gate I^\dagger
            pauli_basis = BasisUnitarySet.get_observable_set(dim=dim)
            for basis_prod in pauli_basis:
                result += (p / 15) * (basis_prod @ (gate @ basis_prod.conj().transpose()))  # P_i gate P_i^\dagger
            return result
        else:
            raise NotImplemented

    def gate_dephase(gate: np.ndarray) -> np.ndarray:
        dim = gate.shape[0]
        result = np.zeros((dim, dim), dtype=complex)
        if dim == 2:
            result += (1 - p) * gate  # I gate I^\dagger
            pauli_basis = list(BasisUnitarySet.get_observable_set(dim=dim))
            result += p * (pauli_basis[3] @ (gate @ pauli_basis[3].conj().transpose()))  # Z gate Z^\dagger
            return result
        elif dim == 4:  # gate_2q_dephase
            result += (1 - p) * gate  # I gate I^\dagger
            pauli_basis = list(BasisUnitarySet.get_observable_set(dim=2))
            pauli_basis = [pauli_basis[0], pauli_basis[3]]  # I, Z
            for i, basis_A in enumerate(pauli_basis):
                for j, basis_B in enumerate(pauli_basis):
                    if i == j == 0:
                        continue
                    basis_prod = np.kron(basis_A, basis_B)
                    result += (p / 3) * (basis_prod @ (gate @ basis_prod.conj().transpose()))  # Z gate Z^\dagger
            return result
        else:
            raise NotImplemented

    def gate_default(gate: np.ndarray) -> np.ndarray:
        return gate

    if name == 'default':
        return gate_default
    elif name == 'depolar':
        return gate_depolarize
    elif name == 'dephase':
        return gate_dephase
    else:
        raise NotImplemented


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
    from QP_QEM_lib import o_tilde_1q, basis_ops, apply_to_qubit, swap_circuit
    from src.circuit_noise_extension import Noisify
    dummy_qubits = cirq.LineQubit.range(2)
    two_qubit_gate = cirq.CNOT
    pure_gate = cirq.H

    cnot_unitary = cirq.unitary(two_qubit_gate)
    pure_unitary = cirq.unitary(pure_gate)

    # --------------------------

    def decomposition_time_test(gate: np.ndarray):
        start_time = time.time()
        qp, basis = BasisUnitarySet.get_decomposition(gate=gate)
        rcs_target = reconstruct_from_basis(qp, basis)
        print(f'Calculation time: {time.time() - start_time} sec.')
        print(f'Sum of quasi-probabilities: {sum(qp)}')
        gst_target = BasisUnitarySet.gate_set_tomography(gate=gate)  # For reference
        print(f'Target vs Reconst. difference: {get_matrix_difference(rcs_target, gst_target)}')
    # decomposition_time_test(gate=pure_unitary)

    # --------------------------

    # Example circuit
    operator_x = cirq.X.on(dummy_qubits[0])
    operator_pure = pure_gate.on(dummy_qubits[0])
    operator_cnot = two_qubit_gate(dummy_qubits[0], dummy_qubits[1])

    p = 0.5  # Depolarizing probability

    def reference_noise(hard_coded: Callable[[np.ndarray], np.ndarray], cirq_noise: cirq.DepolarizingChannel):
        # Initial state density matrix
        init_state = np.array([[1, 0], [0, 0]])  # np.array([[1, 0], [0, 0]])
        # hadamard = cirq.unitary(cirq.H)
        # init_state = hadamard @ (init_state @ hadamard.conj().transpose())
        init_state = np.kron(init_state, init_state)
        print(f'initial state matrix::\n{init_state}\n')

        effective_depolarize = hard_coded(init_state)
        print(f'Custom super operator for depolarizing:\n{effective_depolarize}\n')
        # Effective unitary
        noise_model = INoiseModel(noise_gates=[cirq_noise], description=f'Depolarize (p={cirq_noise.p})')
        effective_unitary = noise_model.get_effective_gate(gate=init_state)
        print(f'Cirq mixture for depolarizing:\n{effective_unitary}\n')

    # reference_noise(get_hardcoded_noise(name='depolar', p=p), cirq.depolarize(p=p))

    # --------------------------

    n = 2
    circuit = swap_circuit(n, cirq.LineQubit.range(2*n+1), True)
    wrapper = get_hardcoded_noise(name='depolar', p=.0)

    # Manager test
    manager = IErrorMitigationManager(clean_circuit=circuit, noise_wrapper=wrapper)
    manager.set_identifier_range(50)
    print(manager)
    mu_values, mu = manager.get_mu_effective()
    print(mu_values)
    print(mu)
    # Plotting for the lol
    import matplotlib.pyplot as plt
    n_bins = 10
    fig, axs = plt.subplots(1, 1, tight_layout=True)

    # We can set the number of bins with the `bins` kwarg
    axs.hist(mu_values,  bins=n_bins)
    plt.show()

    # --------------------------

    # print(circuit)
    # gate_to_qp = IErrorMitigationManager.get_qp_distribution_from_circuit(clean_circuit=circuit, noise_wrapper=wrapper)
    # print(f'Look-up dictionary keys:\n{gate_to_qp.keys()}\nEntire:\n{gate_to_qp}')
    # # Example
    # gate = list(gate_to_qp.keys())[1]
    # qp = gate_to_qp[gate]
    # print(f'Total quasi probability sum: {sum(qp)} for gate {gate}')
    # gate = cirq.unitary(gate)
    # basis = list(BasisUnitarySet.get_basis_set(gate.shape[0]))
    # for i in range(5):
    #     random_basis = IErrorMitigationManager.get_quasi_probability_induced_basis(basis, qp)
    #     print(random_basis)

    # --------------------------

    # noise_model = INoiseModel(noise_gates=[cirq.depolarize(p=p)], description=f'Depolarize (p={p})')
    # noisy_circuit = Noisify.introduce_noise(circuit, noise_model.get_callable())
    # print(f'{noisy_circuit}\n')
    # # print(f'Noise mixture (at p={p}):\n{cirq.mixture(cirq.depolarize(p=p))}')
    # get_qp_distribution_from_circuit(clean_circuit=circuit, noise_model=noise_model)

    # effective_noise_operator = (IBasisOperator(unitary=effective_unitary)).on(dummy_qubits[0])
    # circuit = cirq.Circuit(effective_noise_operator)
    # print(f'{circuit}\n')
    # get_qp_distribution_from_circuit(clean_circuit=circuit, noise_model=None)

    #
    # noise_model_custom = INoiseModel(noise_gates=[gate_depo], description=f'Super Operator Depolarize (e={p})')
    # noisy_circuit_custom = Noisify.introduce_noise(circuit, noise_model_custom.get_callable())
    # print(f'{noisy_circuit_custom}\n')
    # get_qp_distribution_from_circuit(clean_circuit=circuit, noise_model=noise_model_custom)

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
    # gst_target = gate_set_tomography(pure_gate).toarray()
    # gst_basis = [gate_set_tomography(u).toarray() for u in BasisUnitarySet.get_basis_unitary_set()]
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
