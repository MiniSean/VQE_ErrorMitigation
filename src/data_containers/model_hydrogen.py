from typing import Sequence, Tuple, Callable, List
import cirq
import numpy as np
import os
import itertools
from openfermion import MolecularData, QubitOperator, jordan_wigner
from openfermionpsi4 import run_psi4

from src.data_containers.helper_interfaces.i_wave_function import IGeneralizedUCCSD
from src.data_containers.helper_interfaces.i_parameter import IParameter


class HydrogenAnsatz(IGeneralizedUCCSD):

    def __init__(self):
        molecule_params = IParameter({'r0': .7414})  # .7414, 'r1': 1.
        super().__init__(molecule_params)

    def _generate_molecule(self, p: IParameter) -> MolecularData:
        """Produce molecule that can be used by the hamiltonian.
        Using a singlet state with S = 0 to specify we are looking for the lowest singlet energy state.
        multiplicity = 2S + 1
        """
        geometry = [('H', (0., 0., 0.)), ('H', (0., 0., p['r0']))]
        basis = 'sto-3g'
        multiplicity = 1
        charge = 0
        description = str(p['r0'])

        # Change to whatever directory you want
        cwd = os.getcwd()
        data_directory = cwd+'/mol_data'

        if not os.path.exists(data_directory):
            os.mkdir(data_directory)

        filename = data_directory+'/H2_'+description

        run_scf = 1
        run_mp2 = 1
        run_cisd = 1
        run_ccsd = 1
        run_fci = 1
        delete_input = False
        delete_output = False
        verbose = False

        molecule = MolecularData(
            geometry,
            basis,
            multiplicity,
            description=description,
            filename=filename)

        if os.path.exists('{}.hdf5'.format(filename)):
            molecule.load()

        else:
            molecule = run_psi4(molecule,
                                verbose=verbose,
                                run_scf=run_scf,
                                run_mp2=run_mp2,
                                run_cisd=run_cisd,
                                run_ccsd=run_ccsd,
                                run_fci=run_fci)
        # print(f'Reference (Full Configuration Energy: {molecule.fci_energy})')
        return molecule

    # IWaveFunction
    def initial_state(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        """
        Using Hartree Fock state:
        |1100>
        instead of the most general state:
        |0011> + |1100> + |1001> + |0110> (disregarding normalization)

        :param qubits: Circuit qubits, (0, 1, 2, 3)
        :return:  X_0 X_1
        """
        yield [cirq.rx(np.pi).on(qubits[0]),
               cirq.rx(np.pi).on(qubits[1])]

    def observable_measurement(self) -> Tuple[Callable[[cirq.Circuit, Callable[[List[cirq.Qid]], List[cirq.Operation]], int], float], int]:
        """
        Prepares an observable measurement function.
        Not yet general but specific for the H2 Hamiltonian objective.
        :return: Measurement function that: (Input) the circuit, ordered qubits and #measurement repetitions,
        (Output) expectation value. Also returns the circuit cost (number of circuits) required to calculate the expectation value.
        """
        # Hamiltonian observable in the form of qubit operators.
        q_op = jordan_wigner(self.molecule.get_molecular_hamiltonian())
        qubits = self.qubits  # ordered_qubits
        pauli_term_size_lookup = {}
        pauli_weight_lookup = {}
        for term_pauli, term_weight in q_op.terms.items():
            term_size = len(term_pauli)
            if term_size not in pauli_term_size_lookup:
                pauli_term_size_lookup[term_size] = []
            # Set dictionaries
            pauli_term_size_lookup[term_size].append(term_pauli)
            pauli_weight_lookup[term_pauli] = term_weight

        def probability_to_expectation(probability: float, out: int):
            """
            :param probability: Statistical probability [0, 1] corresponding to '0' output measurement.
            :param out: Whether to give information about the '0' or '1' output measurement.
            :return: Scaled measurement probability for either '0' or '1' output measurement in range [-1, 1].
            """
            if out == 0:
                return 2 * probability - 1
            else:
                return 1 - 2 * probability

        def measure_observable(base_circuit: cirq.Circuit, noise_wrapper: Callable[[List[cirq.Qid]], List[cirq.Operation]], meas_reps: int) -> float:  # , ordered_qubits: List[cirq.Qid]
            """
            Constructs multiple quantum circuits to measure the expectation value of a given observable.
            :param base_circuit: Base circuit to measure (can be noisy)
            :param noise_wrapper: Noise wrapper function that generates single or two qubit noise operations
            :param ordered_qubits: Ordered qubits (top to bottom) since Hamiltonian requires consistent qubit ordering.
            :param meas_reps: Number of measurement repetitions for each circuit (higher means more accurate) [1, +inf).
            :return: (Approximation of) Hamiltonian observable expectation value.
            """
            result = 0
            basis_map = IGeneralizedUCCSD.pauli_basis_map()  # Consistent qubit mapping
            labels = ['Z' + str(i) for i in range(len(qubits))]  # Computational measurement labels
            simulator = cirq.DensityMatrixSimulator()
            expectation_lookup = QubitOperator(' ', 1.0)  # Base term

            circuit_01 = base_circuit.copy()  # Single pauli term measurement circuit
            for operator in pauli_term_size_lookup[1]:
                circuit_01.append([(basis_map[new_basis](qubits[qubit_id], 1), noise_wrapper(qubits[qubit_id])) for qubit_id, new_basis in operator])
            circuit_01.append(cirq.measure(q, key=labels[i]) for i, q in enumerate(qubits))
            circuit_01_shots = simulator.run(circuit_01, repetitions=meas_reps)

            # Single pauli terms
            for lb in labels:
                probability = circuit_01_shots.multi_measurement_histogram(keys=[lb])[(0,)] / meas_reps
                expectation = probability_to_expectation(probability, out=0)
                expectation_lookup += QubitOperator(lb, expectation)

            # Two pauli terms
            for operator_labels in itertools.combinations(labels, 2):
                probability = (circuit_01_shots.multi_measurement_histogram(keys=operator_labels)[(0, 0)] +
                               circuit_01_shots.multi_measurement_histogram(keys=operator_labels)[(1, 1)]) / meas_reps
                expectation = probability_to_expectation(probability, out=0)
                expectation_lookup += QubitOperator(' '.join(operator_labels), expectation)

            # Four pauli terms
            for operator in pauli_term_size_lookup[4]:  # Four pauli term measurement circuit
                operator_labels = [new_basis + str(qubit_id) for qubit_id, new_basis in operator]
                # Build circuit with basis transformation and measurement operators
                circuit_02 = base_circuit.copy()
                circuit_02.append([(basis_map[new_basis](qubits[qubit_id], 1), noise_wrapper(qubits[qubit_id])) for qubit_id, new_basis in operator])
                circuit_02.append(cirq.measure(q, key=operator_labels[i]) for i, q in enumerate(qubits))
                # Run measurement
                circuit_02_shots = simulator.run(circuit_02, repetitions=meas_reps)
                # Only equal parities
                probability = (circuit_02_shots.multi_measurement_histogram(keys=operator_labels)[(0, 0, 1, 1)] +
                               circuit_02_shots.multi_measurement_histogram(keys=operator_labels)[(1, 1, 0, 0)] +
                               circuit_02_shots.multi_measurement_histogram(keys=operator_labels)[(0, 1, 1, 0)] +
                               circuit_02_shots.multi_measurement_histogram(keys=operator_labels)[(1, 0, 1, 0)] +
                               circuit_02_shots.multi_measurement_histogram(keys=operator_labels)[(1, 0, 0, 1)] +
                               circuit_02_shots.multi_measurement_histogram(keys=operator_labels)[(0, 1, 0, 1)] +
                               circuit_02_shots.multi_measurement_histogram(keys=operator_labels)[(0, 0, 0, 0)] +
                               circuit_02_shots.multi_measurement_histogram(keys=operator_labels)[(1, 1, 1, 1)]) / meas_reps
                expectation = probability_to_expectation(probability, out=0)
                expectation_lookup += QubitOperator(' '.join(operator_labels), expectation)

            # print(expectation_lookup)

            # Multiply hamiltonian term weights with circuit expectation values for each specific qubit operator set
            for operator in itertools.chain.from_iterable(pauli_term_size_lookup.values()):
                if operator in expectation_lookup.terms:
                    result += pauli_weight_lookup[operator] * expectation_lookup.terms[operator]
                    # print(f'Added {operator}(term): {pauli_weight_lookup[operator]}(weight) * {expectation_lookup.terms[operator]}(expectation) = {pauli_weight_lookup[operator] * expectation_lookup.terms[operator]}')
            return result

        return measure_observable, 5
