# Random parameter sampling from probability distribution
import cirq
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # For displaying for-loop process to console
from typing import Tuple, Union, Callable, Iterator, List, Dict, Sequence, Any, Optional
from src.data_containers.helper_interfaces.i_noise_wrapper import INoiseModel
import itertools as it  # Temp
from multiprocessing import Pool

# WARNING uses multiprocessing
MAX_MULTI_PROCESSING_COUNT = 6


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

    # Get normal unitary set
    @staticmethod
    def get_basis_set(dim: int) -> Iterator[np.ndarray]:
        """Yields all basis unitaries depending on gate dimension"""
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

    # Get T mapping matrix for PTM
    @staticmethod
    def get_t_map(dim: int) -> np.ndarray:
        t_map = np.array([[1, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1], [1, -1, 0, 0]])  # Mapping transform
        if dim == 2:  # (single qubit)
            return t_map
        elif dim == 4:  # (double qubits)
            return np.kron(t_map, t_map)
        else:
            raise NotImplemented

    # Get basis observable set for PTM (single/double qubit gates)
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

    # Get initial state set for PTM (single/double qubit gates)
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

    # Get PTM transformed unitary
    @staticmethod
    def get_ptm_basis_set(dim: int) -> Iterator[np.ndarray]:
        """Yields all ptm transformed basis unitaries depending on gate dimension"""
        def _single_qubit() -> Iterator[np.ndarray]:
            for unitary in BasisUnitarySet.get_basis_unitary_set():
                yield BasisUnitarySet.pauli_transfer_matrix(unitary)

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

    # Apply subsequent noise model to gate before applying this function
    # Refer to figure 4.2 of 'Introduction to Quantum Gate Set Tomography' - Daniel Greenbaum
    @staticmethod
    def pauli_transfer_matrix(gate: np.ndarray, noise_wrapper: Callable[[np.ndarray], np.ndarray] = None) -> np.ndarray:  # Tuple[np.ndarray, np.ndarray]:
        """
        Implements single and double qubit gate PTM.
        Using mapping matrix T:
        [[1, 1, 1, 1],
         [0, 0, 1, 0],
         [0, 0, 0, 1],
         [1, -1, 0, 0]]
        :type gate: Gate (without noise) to apply PTM on
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
        id_list = []  # Order specific identifier collection
        id_sign = []
        id_weight = []
        for operator in circuit.all_operations():
            if type(operator.gate).__name__ == 'MeasurementGate':
                id_weight.append(1)
                continue
            qp = gate_lookup[operator.gate]
            basis_id, weight = QuasiCircuitIdentifier._get_identifier(quasi_probabilities=qp)
            id_list.append(basis_id)  # Append basis basis_id to result list
            id_sign.append(qp[basis_id])  # Multiply probability sign to result sign
            id_weight.append(weight)  # Multiply quasi probability weight
        return id_list, np.sign(np.prod(id_sign)), id_weight

    @staticmethod
    def _get_identifier(quasi_probabilities: List[float]) -> Tuple[int, float]:
        """
        Returns random index from quasi probability list depending on probability distribution and weight
        :param quasi_probabilities: Quasi probabilities that decide the index chosen
        :return: Basis index, total quasi probability weight
        """
        c_weight = sum([abs(p) for p in quasi_probabilities])
        prob_q = np.abs(quasi_probabilities) / c_weight
        quasi_index = np.random.choice(range(len(quasi_probabilities)), p=prob_q)
        return quasi_index, c_weight

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

    def process_circuit(self, process_settings: Tuple[QuasiCircuitIdentifier, int]) -> List[float]:
        _key, _value = process_settings
        # Possible toggle to use error mitigation or not
        _identifier_list = _key.identifiers_id if self._error_mitigation else _key.identity
        _sign_weight = _key.sign if self._error_mitigation else 1
        _weight_list = _key.identifiers_weight if self._error_mitigation else [1]

        # Circuit with noise and noise counter
        _circuit_to_run = self.get_updated_circuit(clean_circuit=self.circuit,
                                                   noise_wrapper=self._noise_model.get_operators,
                                                   circuit_identifier=_identifier_list,
                                                   include_measurements=(not self._density_representation))

        _simulator = cirq.DensityMatrixSimulator(ignore_measurement_results=False)  # Mixed state simulator
        # Repeat measurement for every occurrence of identical circuit identifier
        if self._density_representation:
            # Setup simulator
            _simulated_result = _simulator.simulate(program=_circuit_to_run)  # Include final density matrix

            # Get final density matrix trace
            _rho = np.array(_simulated_result.final_density_matrix)
            _trace_out = np.kron(np.eye(int(_rho.shape[0]/2)), np.array([[1, 0], [0, -1]]))  # Z o I o I o I o I
            if self._hamiltonian_objective is not None:
                _trace_out = self._hamiltonian_objective
            _trace = (_trace_out * _rho).diagonal().sum()
            _measurement = _trace.real
        else:
            # Setup simulator
            _simulated_result = _simulator.run(program=_circuit_to_run, repetitions=(self._meas_reps * _value))  # Include final measurement values

            # Get final measurement value
            _hist = _simulated_result.histogram(key='M')  # Hist (0: x, 1: meas_reps - x)
            _measurement = _hist[0] / (self._meas_reps * _value)  # Normalized measurement [0, 1]
            _measurement = 2 * _measurement - 1  # Rescaled measurement [-1, 1]

        return [_sign_weight * _measurement] * _value

    def get_mu_effective(self, error_mitigation: bool, density_representation: bool, meas_reps: int = 1) -> Tuple[List[float], float]:
        """
        Constructs circuits based on identifiers_id, measures them and calculates effective average.
        :type error_mitigation: Whether to use error mitigation or not.
        :param density_representation: Whether to use density matrix simulation or not.
        :param meas_reps: Amount of circuit measurements performed when density_representation is false.
        :return: List of (signed) measurement values, Effective average of circuit measurement values.
        """
        # TEMP variable setting
        self._error_mitigation = error_mitigation
        self._density_representation = density_representation
        self._meas_reps = meas_reps

        # Determine if it is advantageous to use multiprocessing
        item_count = len(self._dict_identifiers)
        process_iterator = tqdm(self._dict_identifiers.items(), desc='Processing identifier circuits')
        processing_output = list()
        if item_count < MAX_MULTI_PROCESSING_COUNT:
            # Use normal iteration
            for item in process_iterator:
                processing_output.append(self.process_circuit(item))
        else:
            # Use multiprocessing
            # Function for multiprocessing
            with Pool(MAX_MULTI_PROCESSING_COUNT) as p:
                processing_output = p.map(self.process_circuit, process_iterator)

        def flatten(l: List[List[Any]]) -> np.ndarray:
            return np.array([item for sublist in l for item in sublist])
        raw_measurement_results = flatten(processing_output)  # Flatten output

        weight_list = next(iter(self._dict_identifiers)).identifiers_weight if error_mitigation else [1]  # All identifier weights are equal
        weighted_effective_mean = np.prod(weight_list) * np.mean(raw_measurement_results)
        return np.prod(weight_list) * raw_measurement_results, weighted_effective_mean

    # Temp
    def get_identifiers(self) -> Iterator[List[int]]:
        for id_class in self._dict_identifiers.keys():
            yield id_class.identifiers

    def __init__(self, clean_circuit: cirq.Circuit, noise_model: INoiseModel, hamiltonian_objective: Optional[np.ndarray]):
        # Store properties
        self.circuit = clean_circuit
        self._noise_model = noise_model
        self._gate_lookup = IErrorMitigationManager.get_qp_lookup(clean_circuit, noise_model.get_effective_gate)
        self._dict_identifiers = {}
        # TEMP
        self._hamiltonian_objective = hamiltonian_objective
        self._error_mitigation = False
        self._density_representation = True
        self._meas_reps = 1

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
            clean_ptm_gate = BasisUnitarySet.pauli_transfer_matrix(clean_gate)
            noisy_ptm_gate = BasisUnitarySet.pauli_transfer_matrix(clean_gate, noise_wrapper=noise_wrapper)
            # Get inverse noise
            N_inv = clean_ptm_gate.dot(np.linalg.inv(noisy_ptm_gate))  # Inverse method

            basis_set = list(BasisUnitarySet.get_ptm_basis_set(dim=clean_gate.shape[0]))
            qp = BasisUnitarySet.get_quasiprobabilities(target=N_inv, basis_set=basis_set)
            result[operator.gate] = qp
        return result

    @staticmethod
    def get_updated_circuit(clean_circuit: cirq.Circuit, noise_wrapper: Callable[[Tuple[cirq.Qid]], List[cirq.Operation]], circuit_identifier: List[int], include_measurements: bool = False) -> cirq.Circuit:  # QuasiCircuitIdentifier
        """
        Constructs circuit based on noiseless (clean) circuit.
        After each operation a noise operation and noise mitigation operation is placed.
        If the noise mitigation operator is not the identity operator an additional noise operation is placed afterwards.
        :param clean_circuit: Noiseless circuit used as basis.
        :param noise_wrapper: Function that provides noise channel depending on number of qubits it acts on.
        :param circuit_identifier: Pre-calculated identifiers that points to a basis operation (used as mitigation) from known set.
        :param include_measurements: Whether to include measurement operations (relevant when using density matrix simulators).
        :return:
        """
        result = cirq.Circuit()
        for i, clean_operator in enumerate(clean_circuit.all_operations()):
            if type(clean_operator.gate).__name__ == 'MeasurementGate':  # Bail on measurement operator
                if include_measurements:
                    result.append(clean_operator)
                continue

            # First Ideal operator
            result.append(clean_operator)

            # Get clean gate unitary
            unitary = cirq.unitary(clean_operator.gate)
            # Use correct basis
            basis_dim = unitary.shape[0]
            basis_set = list(BasisUnitarySet.get_basis_set(basis_dim))

            # Build noisy and canceling basis
            noise_mitigating_basis = basis_set[circuit_identifier[i]]
            # Construct operators
            qubit_list = clean_operator.qubits
            noisy_operator = noise_wrapper(*qubit_list)  # qubit_list[0]
            if len(qubit_list) == 1:
                noise_canceling_operator = IBasisGateSingle(unitary=noise_mitigating_basis).on(*qubit_list)  # qubit_list[0]
            elif len(qubit_list) == 2:
                noise_canceling_operator = IBasisGateTwo(unitary=noise_mitigating_basis).on(*qubit_list)  # qubit_list[0], qubit_list[1]
            else:
                raise NotImplemented('Can only construct single and two qubit operators')
            # Add to new circuit
            result.append(noisy_operator)  # Noise after ideal operation
            result.append(noise_canceling_operator)  # Correction after noise
            if circuit_identifier[i] != 0:  # Only if correction basis is not identity
                result.append(noisy_operator)  # Noise after correction
        return result

    def __str__(self):
        return self._dict_identifiers.__str__()


def reconstruct_from_basis(quasi_probabilities: List[float], basis_set: List[np.ndarray]) -> np.ndarray:
    """Reverse function of solving the linear set of equations"""
    if len(quasi_probabilities) != len(basis_set):
        raise ValueError(f'Each basis should have a corresponding quasi-probability value')
    return sum(i[0] * i[1] for i in zip(quasi_probabilities, basis_set))  # Dot product of lists


def simulate_error_mitigation(clean_circuit: cirq.Circuit, noise_model: INoiseModel, process_circuit_count: int, density_representation: bool, desc: str, hamiltonian_objective: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
    """
    Simulates error mitigation on noise infected circuit.
    The four subplots consists of:
    (1) Ideal circuit,
    (2) Ideal with error mitigation,
    (3) Ideal with noise channels,
    (4) Ideal with noise channels and error mitigation.
    :param clean_circuit: Noiseless quantum circuit to evaluate
    :param noise_model: Noise model that holds information about the applied noise channels
    :param process_circuit_count: Number of circuit identifiers produced to handle error mitigation
    :type hamiltonian_objective: Enables to interpret final density matrix by hamiltonian objective
    :return: None
    """
    # Settings
    noise_probability = [False, True]
    noise_identifier_count = process_circuit_count
    circuit_measurement_repetitions = 1
    using_error_mitigation = [False, True]
    using_density_representation = density_representation if hamiltonian_objective is None else True
    n_bins = 100
    fig, axs = plt.subplots(2, 2, tight_layout=True)
    # Construct circuit
    circuit = clean_circuit
    circuit_name = f'{desc} using {"density matrix" if using_density_representation else "measurements"}'
    fig.suptitle(f'{circuit_name} (#mitigation circuits={noise_identifier_count})', fontsize=16)

    mu_ideal = None
    mu_noisy = None
    mu_effective = None

    for i, prob in enumerate(noise_probability):
        for j, error in enumerate(using_error_mitigation):
            # Toggle noise
            _noise_model = noise_model if prob else INoiseModel.empty()
            # Error Mitigation Manager
            manager = IErrorMitigationManager(clean_circuit=circuit, noise_model=_noise_model, hamiltonian_objective=hamiltonian_objective)
            # Calculate measurement values
            manager.set_identifier_range(noise_identifier_count)
            meas_values, mu = manager.get_mu_effective(error_mitigation=error,
                                                       density_representation=using_density_representation,
                                                       meas_reps=circuit_measurement_repetitions)
            # Store ideal
            if i == j == 0:
                mu_ideal = mu
            # Store effective
            if i == 1 and j == 0:
                mu_noisy = mu
            # Store effective
            if i == j == 1:
                mu_effective = mu

            # Plotting
            show_meas_reps = '' if using_density_representation else f', reps={circuit_measurement_repetitions}'
            plot_title = f'measurement histogram.\n(Info: {_noise_model.get_description()}, error mitigation={error}{show_meas_reps})'
            axs[i][j].title.set_text(plot_title)
            x_lim = [int(mu_ideal - 1), int(mu_ideal + 1)] if hamiltonian_objective is not None else [-1, 1]
            axs[i][j].set_xlabel(f'relative measurement [{x_lim[0]}, {x_lim[1]}]')  # X axis label
            axs[i][j].set_xlim(x_lim)  # X axis limit
            axs[i][j].set_ylabel(f'Bin count')  # Y axis label
            axs[i][j].hist(meas_values, bins=n_bins)
            if i != 0:
                axs[i][j].axvline(x=mu, linewidth=1, color='r', label=r'C E[$\mu_{eff}$] = ' + f'{np.round(mu, 5)}\n' + r'|$\epsilon$| = ' + f'{np.round(abs(mu - mu_ideal), 5)}')
            axs[i][j].axvline(x=mu_ideal, linewidth=1, color='g', label=r'$\mu_{ideal}$ = ' + f'{np.round(mu_ideal, 5)}')
            axs[i][j].legend()
    return mu_ideal, mu_noisy, mu_effective


# Deprecated
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
    # Use correct basis
    basis_dim = gate.shape[0]
    basis_set = list(BasisUnitarySet.get_basis_set(basis_dim))
    qp = BasisUnitarySet.get_quasiprobabilities(target=gate, basis_set=basis_set)
    rcs_target = reconstruct_from_basis(qp, basis_set)
    print(f'Calculation time: {time.time() - start_time} sec.')
    print(f'Sum of quasi-probabilities: {sum(qp)}')
    ptm_target = BasisUnitarySet.pauli_transfer_matrix(gate=gate)  # For reference
    print(f'Target vs Reconst. difference: {get_matrix_difference(rcs_target, ptm_target)}')


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
                                  cirq.protocols.unitary(cirq.ops.pauli_gates.X),
                                  cirq.protocols.unitary(cirq.ops.pauli_gates.Y),
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


class SingleQubitPauliChannel(cirq.SingleQubitGate):
    def __init__(self, p_x: float, p_y: float, p_z: float) -> None:
        self._p_i = 1 - p_x - p_y - p_z
        self._p_x = p_x
        self._p_y = p_y
        self._p_z = p_z

    def _mixture_(self) -> Sequence[Tuple[float, np.ndarray]]:
        result = ((self._p_i, cirq.unitary(cirq.I)),
                  (self._p_x, cirq.unitary(cirq.X)),
                  (self._p_y, cirq.unitary(cirq.Y)),
                  (self._p_z, cirq.unitary(cirq.Z)))
        return result

    @staticmethod
    def _has_mixture_() -> bool:
        return True

    def __repr__(self) -> str:
        return f'cirq.single_qubit_asymmetry_depolarize(p_x={self._p_x}, p_y={self._p_y}, p_z={self._p_z})'

    def __str__(self) -> str:
        return f'single_qubit_asymmetry_depolarize(p_x={self._p_x}, p_y={self._p_y}, p_z={self._p_z})'

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs' ) -> 'cirq.CircuitDiagramInfo':
        return cirq.protocols.CircuitDiagramInfo(wire_symbols=('D({:.3f})'.format(self._p_i),))


class TwoQubitPauliChannel(cirq.TwoQubitGate):
    def __init__(self, p_x: float, p_y: float, p_z: float) -> None:
        self._p_i = 1 - p_x - p_y - p_z
        self._p_x = p_x
        self._p_y = p_y
        self._p_z = p_z

    def _single_mixture_(self) -> Sequence[Tuple[float, np.ndarray]]:
        result = ((self._p_i, cirq.unitary(cirq.I)),
                  (self._p_x, cirq.unitary(cirq.X)),
                  (self._p_y, cirq.unitary(cirq.Y)),
                  (self._p_z, cirq.unitary(cirq.Z)))
        return result

    def _mixture_(self) -> Sequence[Tuple[float, np.ndarray]]:
        result = ()
        for p_A, matrix_A in self._single_mixture_():
            for p_B, matrix_B in self._single_mixture_():
                result += ((p_A * p_B, np.kron(matrix_A, matrix_B)),)
        return result

    @staticmethod
    def _has_mixture_() -> bool:
        return True

    def __repr__(self) -> str:
        return f'cirq.two_qubit_asymmetry_depolarize(p_x={self._p_x}, p_y={self._p_y}, p_z={self._p_z})'

    def __str__(self) -> str:
        return f'two_qubit_asymmetry_depolarize(p_x={self._p_x}, p_y={self._p_y}, p_z={self._p_z})'

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs' ) -> 'cirq.CircuitDiagramInfo':
        return cirq.protocols.CircuitDiagramInfo(wire_symbols=('D({:.3f})'.format(self._p_i), 'D({:.3f})'.format(self._p_i)))


class SingleQubitLeakageChannel(cirq.SingleQubitGate):
    def __init__(self, p: float) -> None:
        self._p = p

    def _mixture_(self) -> Sequence[Tuple[float, np.ndarray]]:
        result = ((1, np.array([[1, 0], [0, 0]])),
                  (np.sqrt(1 - self._p), np.array([[0, 0], [0, 1]])))
        return result

    @staticmethod
    def _has_mixture_() -> bool:
        return True

    def __repr__(self) -> str:
        return f'cirq.single_qubit_leakage(p={self._p})'

    def __str__(self) -> str:
        return f'single_qubit_leakage(p={self._p})'

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs' ) -> 'cirq.CircuitDiagramInfo':
        return cirq.protocols.CircuitDiagramInfo(wire_symbols=('L({:.3f})'.format(self._p),))


class TwoQubitLeakageChannel(cirq.TwoQubitGate):
    def __init__(self, p: float) -> None:
        self._p = p

    def _single_mixture_(self) -> Sequence[Tuple[float, np.ndarray]]:
        result = ((1, np.array([[1, 0], [0, 0]])),
                  (np.sqrt(1 - self._p), np.array([[0, 0], [0, 1]])))
        return result

    def _mixture_(self) -> Sequence[Tuple[float, np.ndarray]]:
        result = ()
        for p_A, matrix_A in self._single_mixture_():
            for p_B, matrix_B in self._single_mixture_():
                result += ((p_A * p_B, np.kron(matrix_A, matrix_B)),)
        return result

    @staticmethod
    def _has_mixture_() -> bool:
        return True

    def __repr__(self) -> str:
        return f'cirq.two_qubit_leakage(p={self._p})'

    def __str__(self) -> str:
        return f'two_qubit_leakage(p={self._p})'

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs' ) -> 'cirq.CircuitDiagramInfo':
        return cirq.protocols.CircuitDiagramInfo(wire_symbols=('L({:.3f})'.format(self._p), 'L({:.3f})'.format(self._p)))


if __name__ == '__main__':
    import time
    from QP_QEM_lib import o_tilde_1q, basis_ops, apply_to_qubit, swap_circuit, noise_model
    from src.circuit_noise_extension import Noisify
    dummy_qubits = cirq.LineQubit.range(2)

    # Example circuit
    prob = 1e-4
    circuit_01 = cirq.Circuit()
    circuit_02 = cirq.Circuit()
    n_gate = noise_model(dim=2, error_type='depolarize', error=prob)
    channel_1q = [SingleQubitPauliChannel(p_x=1e-4, p_y=1e-4, p_z=6e-4), SingleQubitLeakageChannel(p=8e-4)]
    channel_2q = [TwoQubitPauliChannel(p_x=1e-4, p_y=1e-4, p_z=6e-4), TwoQubitLeakageChannel(p=8e-4)]  # [TwoQubitDepolarizingChannel(p=prob)]
    n_model = INoiseModel(noise_gates_1q=channel_1q, noise_gates_2q=channel_2q, description=f'depolarize (p={prob}')

    circuit_01.append(n_gate.on(*dummy_qubits))
    circuit_02.append(n_model.get_operators(*dummy_qubits))

    density_01 = cirq.DensityMatrixSimulator().simulate(circuit_01).final_density_matrix
    density_02 = cirq.DensityMatrixSimulator().simulate(circuit_02).final_density_matrix

    print(f'Following density matrices should be identical:\n{density_01}\n{density_02}\n')

    # --------------------------

    # Settings
    prob = 1e-4
    n = 2
    # Construct circuit
    circuit = swap_circuit(n, cirq.LineQubit.range(2 * n + 1), True)
    # Construct noise wrapper
    channel_1q = [SingleQubitPauliChannel(p_x=prob, p_y=prob, p_z=6 * prob)]  # , SingleQubitLeakageChannel(p=8 * prob)]
    channel_2q = [TwoQubitPauliChannel(p_x=prob, p_y=prob, p_z=6 * prob)]  # , TwoQubitLeakageChannel(p=8 * prob)]
    noise_model = INoiseModel(noise_gates_1q=channel_1q, noise_gates_2q=channel_2q,
                              description=f'asymmetric depolarization (p={prob})')

    # Plot error mitigation
    mu_ideal, mu_noisy, mu_effective = simulate_error_mitigation(clean_circuit=circuit, noise_model=noise_model, process_circuit_count=1000, density_representation=True, desc='Swap-Circuit')
    print(f'Operator expectation value (noisy): {mu_noisy}\nOperator expectation value (mitigated): {mu_effective}\nFinal difference error: {abs(mu_ideal - mu_effective)}')
    plt.show()
