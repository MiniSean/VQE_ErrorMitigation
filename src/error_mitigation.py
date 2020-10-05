# Random parameter sampling from probability distribution
import random
import cirq
import numpy as np
from tqdm import tqdm  # For displaying for-loop process to console
from typing import Tuple, Union, Callable, Iterator, List, Dict, Sequence
from src.data_containers.helper_interfaces.i_noise_wrapper import INoiseModel
import itertools as it  # Temp


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

    # Get T mapping matrix for GST
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
        for j, obs in enumerate(_observable_set):
            for k, rho in enumerate(_initial_state_set_ptm):
                ptm_estimation[j, k] = np.trace(obs @ rho)  # Trace
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
        self.identifiers_id, self.sign, self.identifiers_weight = QuasiCircuitIdentifier._get_identifiers(circuit=circuit, gate_lookup=gate_lookup)

    @staticmethod
    def _get_identifiers(circuit: cirq.Circuit, gate_lookup: Dict[cirq.Gate, List[float]]) -> Tuple[List[int], int, List[float]]:
        id_list = list()  # Order specific identifier collection
        id_sign = 1
        id_weight = []
        for operator in circuit.all_operations():
            if type(operator.gate).__name__ == 'MeasurementGate':
                id_weight.append(1)
                continue
            qp = gate_lookup[operator.gate]
            basis_id, p_sign, weight = QuasiCircuitIdentifier._get_identifier(quasi_probabilities=qp)
            id_list.append(basis_id)  # Append basis basis_id to result list
            # np.sign(p_sign) == np.sign(qp[basis_id])
            id_sign *= p_sign  # Multiply probability sign to result sign
            id_weight.append(weight)  # Multiply quasi probability weight
        return id_list, id_sign, id_weight

    @staticmethod
    def _get_identifier(quasi_probabilities: List[float]) -> Tuple[int, int, float]:
        """
        Returns random index from quasi probability list depending on probability distribution, its sign and weight
        :param quasi_probabilities:
        :return: Basis index, quasi probability sign, total quasi probability weight
        """
        c_weight = sum([abs(p) for p in quasi_probabilities])
        p_indicator = random.uniform(0, 1) * c_weight  # Scales random from [0, c_weight]
        for i, probability in enumerate(quasi_probabilities):
            p_indicator -= abs(probability)
            if p_indicator <= 0:
                return i, np.sign(probability), c_weight
        last_index = len(quasi_probabilities) - 1
        return last_index, np.sign(quasi_probabilities[last_index]), c_weight

    @property
    def identity(self):
        """Returns list of 0's with size of len(self.identifiers_id) targeting only identity basis matrices"""
        return [0 for basis_id in range(len(self.identifiers_id))]

    def __eq__(self, other):
        return other.__hash__() == self.__hash__()

    def __hash__(self):
        return hash(tuple(self.identifiers_id))


class IErrorMitigationManager:
    """Manages family of circuits to be measured"""

    def get_mu_effective(self, error_mitigation: bool, density_representation: bool, meas_reps: int = 1) -> Tuple[List[float], float]:
        """
        Constructs circuits based on identifiers_id, measures them and calculates effective average.
        :type error_mitigation: Whether to use error mitigation or not.
        :param density_representation: Whether to use density matrix simulation or not.
        :param meas_reps: Amount of circuit measurements performed when density_representation is false.
        :return: List of (signed) measurement values, Effective average of circuit measurement values.
        """
        raw_measurement_results = list()
        weight_list = [1]
        simulator = cirq.DensityMatrixSimulator(ignore_measurement_results=True)  # Mixed state simulator
        for key, value in tqdm(self._dict_identifiers.items(), desc='Processing identifier circuits'):
            # Possible toggle to use error mitigation or not
            identifier_list = key.identifiers_id if error_mitigation else key.identity
            print(identifier_list)
            sign_weight = key.sign if error_mitigation else 1
            weight_list = key.identifiers_weight if error_mitigation else [1]

            # Circuit with noise and noise counter
            circuit_to_run = IErrorMitigationManager.get_updated_circuit(clean_circuit=self.circuit, noise_wrapper=self._noise_model.get_operators, circuit_identifier=identifier_list)

            # Repeat measurement for every occurrence of identical circuit identifier
            signed_measurement_results = []
            for identifier_occurrence in range(value):
                if density_representation:
                    # Setup simulator
                    simulated_result = simulator.simulate(program=circuit_to_run)  # Include final density matrix

                    # Get final density matrix trace
                    rho = np.array([[1, 0], [0, 0]])
                    identity = np.eye(16)
                    trace_out = np.kron(identity, rho)
                    trace = (trace_out @ np.array(simulated_result.final_density_matrix)).diagonal().sum()
                    measurement = trace.real
                else:
                    # Setup simulator
                    simulated_result = simulator.run(program=circuit_to_run, repetitions=meas_reps)  # Include final measurement values

                    # Get final measurement value
                    hist = simulated_result.histogram(key='M')  # Hist (0: x, 1: meas_reps - x)
                    measurement = hist[0]/meas_reps  # Normalized measurement [0, 1]

                # Add individual measurements
                measurement = 2 * measurement - 1  # Rescaled measurement [-1, 1]
                signed_measurement_results.append(sign_weight * measurement)

            raw_measurement_results.append(np.mean(signed_measurement_results))  # Additional averaging step

        weighted_effective_mean = np.prod(weight_list) * np.mean(raw_measurement_results)
        return raw_measurement_results, weighted_effective_mean

    # Temp
    def get_identifiers(self) -> Iterator[List[int]]:
        for id_class in self._dict_identifiers.keys():
            yield id_class.identifiers

    def __init__(self, clean_circuit: cirq.Circuit, noise_model: INoiseModel):  # Callable[[np.ndarray], np.ndarray]):
        # Store properties
        self.circuit = clean_circuit
        self._noise_model = noise_model
        self._gate_lookup = IErrorMitigationManager.get_qp_lookup(clean_circuit, noise_model.get_effective_gate)
        self._dict_identifiers = {}

    # Magic function, adds range of 'QuasiCircuitIdentifier' to dictionary of identifiers_id
    def set_identifier_range(self, count: int):
        self._dict_identifiers = {}
        for i in range(count):
            self.add_circuit_identifier()

    # Magic function, adds 'QuasiCircuitIdentifier' to dictionary of identifiers_id
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
        :param noise_wrapper: Applies noise channel to single or two qubit gate operation
        :return: Dictionary containing gate types as key and both the q-p list as their combined weight as values
        """
        result = {}  # gate to quasi probabilities
        for operator in clean_circuit.all_operations():
            if type(operator.gate).__name__ == 'MeasurementGate' or operator.gate in result.keys():
                continue
            # Get effective gates
            clean_gate = cirq.unitary(operator)  # Without noise
            # Get gate set tomography versions
            clean_gst_gate = BasisUnitarySet.gate_set_tomography(clean_gate)
            noisy_gst_gate = BasisUnitarySet.gate_set_tomography(clean_gate, noise_wrapper=noise_wrapper)
            # Get inverse noise
            N_inv = clean_gst_gate.dot(np.linalg.inv(noisy_gst_gate))

            basis_set = list(BasisUnitarySet.get_gst_basis_set(dim=clean_gate.shape[0]))
            qp = BasisUnitarySet.get_quasiprobabilities(target=N_inv, basis_set=basis_set)
            result[operator.gate] = qp
        return result

    # Deprecated
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
    def get_updated_circuit(clean_circuit: cirq.Circuit, noise_wrapper: Callable[[Tuple[cirq.Qid]], List[cirq.Operation]], circuit_identifier: List[int]) -> cirq.Circuit:  # QuasiCircuitIdentifier
        result = cirq.Circuit()
        for i, clean_operator in enumerate(clean_circuit.all_operations()):
            # First Ideal operator
            result.append(clean_operator)

            if type(clean_operator.gate).__name__ == 'MeasurementGate':
                continue

            # Get clean gate unitary
            unitary = cirq.unitary(clean_operator.gate)
            # Use correct basis
            basis_dim = unitary.shape[0]
            basis_set = list(BasisUnitarySet.get_basis_set(basis_dim))

            # Build noisy and canceling basis
            noise_canceling_basis = basis_set[circuit_identifier[i]]
            # Construct operators
            qubit_list = clean_operator.qubits
            noisy_operator = noise_wrapper(*qubit_list)  # qubit_list[0]
            if len(qubit_list) == 1:
                noise_canceling_operator = IBasisGateSingle(unitary=noise_canceling_basis).on(*qubit_list)  # qubit_list[0]
            elif len(qubit_list) == 2:
                noise_canceling_operator = IBasisGateTwo(unitary=noise_canceling_basis).on(*qubit_list)  # qubit_list[0], qubit_list[1]
            else:
                raise NotImplemented('Can only construct single and two qubit operators')
            # Add to new circuit
            result.append(noisy_operator)  # Noise after ideal operation
            result.append(noise_canceling_operator)  # Correction after noise
            result.append(noisy_operator)  # Noise after correction
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


def decomposition_time_test(gate: np.ndarray):
    start_time = time.time()
    qp, basis = get_decomposition(gate=gate)
    rcs_target = reconstruct_from_basis(qp, basis)
    print(f'Calculation time: {time.time() - start_time} sec.')
    print(f'Sum of quasi-probabilities: {sum(qp)}')
    gst_target = BasisUnitarySet.gate_set_tomography(gate=gate)  # For reference
    print(f'Target vs Reconst. difference: {get_matrix_difference(rcs_target, gst_target)}')


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
    noise_model = INoiseModel(noise_gates_1q=[cirq_noise], description=f'Depolarize (p={cirq_noise.p})')
    effective_unitary = noise_model.get_effective_gate(gate=init_state)
    print(f'Cirq mixture for depolarizing:\n{effective_unitary}\n')


# Hard coded copy of two qubit depolarizing channel
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

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs' ) -> 'cirq.CircuitDiagramInfo':
        return cirq.protocols.CircuitDiagramInfo(
            wire_symbols=('D({:.3f})'.format(self._p), 'D({:.3f})'.format(self._p)))

    @property
    def p_(self) -> float:
        """The probability that an error occurs."""
        return self._p

    # def _json_dict_(self) -> Dict[str, Any]:
    #     return protocols.obj_to_dict_helper(self, ['p'])


if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt
    from QP_QEM_lib import o_tilde_1q, basis_ops, apply_to_qubit, swap_circuit, noise_model
    from src.circuit_noise_extension import Noisify
    dummy_qubits = cirq.LineQubit.range(2)

    # Example circuit
    prob = 1e-2
    circuit_01 = cirq.Circuit()
    circuit_02 = cirq.Circuit()
    n_gate = noise_model(dim=2, error_type='depolarize', error=prob)
    channel_1q = [cirq.depolarize(p=prob)]
    channel_2q = [TwoQubitDepolarizingChannel(p=prob)]
    n_model = INoiseModel(noise_gates_1q=channel_1q, noise_gates_2q=channel_2q, description=f'depolarize (p={prob}')

    circuit_01.append(n_gate.on(*dummy_qubits))
    circuit_02.append(n_model.get_operators(*dummy_qubits))

    density_01 = cirq.DensityMatrixSimulator().simulate(circuit_01).final_density_matrix
    density_02 = cirq.DensityMatrixSimulator().simulate(circuit_02).final_density_matrix

    print(f'Following density matrices should be identical:\n{density_01}\n{density_02}\n')

    # --------------------------

    def plot_single():
        # Settings
        n = 2
        circuit_name = 'Swap-circuit'
        noise_name = 'depolar'
        noise_probability = 1e-2
        noise_identifier_count = 50
        circuit_measurement_repetitions = 1000
        using_error_mitigation = True
        using_density_representation = True
        plot_title = f'{circuit_name} measurement histogram.\n(Info: {noise_name}, p={noise_probability}, error mitigation={using_error_mitigation}, reps={circuit_measurement_repetitions})'

        # Construct circuit and noise wrapper
        circuit = swap_circuit(n, cirq.LineQubit.range(2 * n + 1), True)
        channel_1q = [cirq.depolarize(p=noise_probability)]
        channel_2q = [TwoQubitDepolarizingChannel(p=noise_probability)]
        n_model = INoiseModel(noise_gates_1q=channel_1q, noise_gates_2q=channel_2q, description=f'depolarize (p={noise_probability})')
        print(f'Circuit used:')
        print(circuit)

        # Error Mitigation Manager
        manager = IErrorMitigationManager(clean_circuit=circuit, noise_model=n_model)
        manager.set_identifier_range(noise_identifier_count)
        print(f'Start calculating raw mu from circuit identifiers_id')
        mu_values, mu = manager.get_mu_effective(error_mitigation=using_error_mitigation, density_representation=using_density_representation, meas_reps=circuit_measurement_repetitions)
        # Plotting for the lol
        n_bins = 100
        fig, ax = plt.subplots(1, 1, tight_layout=True)
        ax.title.set_text(plot_title)
        ax.set_xlabel(f'{circuit_name} measurement outcome (avg: {mu})')  # X axis label
        ax.set_xlim([-1, 1])  # X axis limit
        ax.set_ylabel(f'Bin count')  # Y axis label
        # We can set the number of bins with the `bins` kwarg
        ax.hist(mu_values, bins=n_bins)

    def plot_shihonage():
        # Settings
        n = 2
        circuit_name = 'Swap-circuit'
        noise_name = 'depolar'
        noise_probability = [0, 1e-2]
        noise_identifier_count = 50
        circuit_measurement_repetitions = 1000
        using_error_mitigation = [False, True]
        using_density_representation = True
        n_bins = 100
        fig, axs = plt.subplots(2, 2, tight_layout=True)
        # Construct circuit
        circuit = swap_circuit(n, cirq.LineQubit.range(2 * n + 1), True)
        fig.suptitle(f'{circuit_name} (#register qubits={n})', fontsize=16)
        print(circuit)

        for i, prob in enumerate(noise_probability):
            for j, error in enumerate(using_error_mitigation):
                # if i + j != 2:  # Temp
                #     continue
                # Construct noise wrapper
                channel_1q = [cirq.depolarize(p=prob)]
                channel_2q = [TwoQubitDepolarizingChannel(p=prob)]
                noise_model = INoiseModel(noise_gates_1q=channel_1q, noise_gates_2q=channel_2q, description=f'depolarize (p={prob})')
                # Error Mitigation Manager
                manager = IErrorMitigationManager(clean_circuit=circuit, noise_model=noise_model)
                # Calculate measurement values
                manager.set_identifier_range(noise_identifier_count)
                meas_values, mu = manager.get_mu_effective(error_mitigation=error, density_representation=using_density_representation, meas_reps=circuit_measurement_repetitions)
                plot_title = f'{circuit_name} measurement histogram.\n(Info: {noise_name}, p={prob}, error mitigation={error}, reps={circuit_measurement_repetitions})'
                axs[i][j].title.set_text(plot_title)
                axs[i][j].set_xlabel(f'(weighted) measurement outcome (avg: {mu})')  # X axis label
                axs[i][j].set_xlim([-1, 1])  # X axis limit
                axs[i][j].set_ylabel(f'Bin count')  # Y axis label
                axs[i][j].hist(meas_values,  bins=n_bins)

    plot_shihonage()

    plt.show()
