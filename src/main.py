import cirq
import scipy
import numpy as np
from openfermion import MolecularData

from src.data_containers.model_hydrogen import HydrogenAnsatz
from src.processors.processor_quantum import QPU
from src.processors.processor_classic import CPU
import openfermioncirq
import openfermion


if __name__ == '__main__':

    print("H2-Molecule example. Finding expectation value for molecule with optimal atom distance (.7414 angstrom)")
    ansatz = HydrogenAnsatz()
    parameters = ansatz.operator_parameters
    operators = ansatz.operations(ansatz.qubits)
    # print(list(ansatz.operations(ansatz.qubits)))

    # _hamiltonian = ansatz.molecule.get_molecular_hamiltonian()
    # _qubits = count_qubits(_hamiltonian)
    # _trotter = openfermioncirq.simulate_trotter(qubits=_qubits, hamiltonian=_hamiltonian, time=1.0, n_steps=1, order=0, algorithm=openfermioncirq.trotter.LOW_RANK, omit_final_swaps=True)
    # _circuit = cirq.Circuit(_trotter)
    # cirq.merge_single_qubit_gates_into_phased_x_z(_circuit)
    # print(_circuit.to_text_diagram(use_unicode_characters=False))

    # # Hartree Fock state
    # hf_state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])  # 4^2 = 16 possibilities
    #
    # # second quantization hamiltonian
    # _hamiltonian = ansatz.molecule.get_molecular_hamiltonian()
    # _qubits = openfermion.count_qubits(_hamiltonian)
    # # UCC-Single amplitudes
    # _single_amp = ansatz.molecule.ccsd_single_amps
    # # UCC-Double amplitudes
    # _double_amp = ansatz.molecule.ccsd_double_amps
    # # Creation/Annihilation operators
    # _ucc_operator = openfermion.normal_ordered(openfermion.uccsd_generator(_single_amp, _double_amp))
    # # Pauli string operators
    # _ucc_qubitop = openfermion.jordan_wigner(_ucc_operator)
    #
    # for i, fo in enumerate([_ucc_operator][::2]):
    #     # Jordan-Wigner transform each fermionic operator + its Hermitian conjugate
    #     qo_list = list(openfermion.jordan_wigner(fo + openfermion.hermitian_conjugated(fo)))
    #     print(qo_list)
    #
    # # # Caculate energy using matrix multiplication without trotter
    # # _ham_arr = openfermion.get_sparse_operator(openfermion.transforms.jordan_wigner(_hamiltonian)).toarray()
    # # _ucc_sparse = (openfermion.get_sparse_operator(_ucc_qubitop, n_qubits=4))
    # # _ucc_exp = scipy.linalg.expm(_ucc_sparse)
    # # _ucc_state = (np.dot(_ucc_exp.toarray(), hf_state))
    #
    # print(_ucc_operator)
    # print(_ucc_qubitop)

    # Get resolved circuit
    circuit = ansatz.circuit
    print(circuit)

    # Get variational study
    result = CPU.get_optimized_state(w=ansatz, max_iter=10)
    print(f'Operator expectation value: {result.optimal_value}\nOperator parameters: {result.optimal_parameters}')

    parameters.update(r=result)
    resolved_circuit = QPU.get_resolved_circuit(circuit, parameters)
    print(resolved_circuit)

    # --------------------------

    from src.data_containers.helper_interfaces.i_wave_function import IGeneralizedUCCSD
    from src.data_containers.helper_interfaces.i_parameter import IParameter


    class CCSD_Ansatz(IGeneralizedUCCSD):

        def __init__(self):
            operator = IParameter({'alpha': 1.})
            molecule = IParameter({'r0': .7414})  # .7414
            super().__init__(operator, molecule)

        # IWaveFunction
        def _generate_molecule(self, p: IParameter) -> MolecularData:
            """Produce molecule that can be used by the hamiltonian.
            Using a singlet state with S = 0 to specify we are looking for the lowest singlet energy state.
            multiplicity = 2S + 1
            """
            r = p['r0']
            geometry = [('H', (0., 0., 0.)), ('H', (0., 0., r))]
            molecule = MolecularData(geometry=geometry, basis='sto-3g', multiplicity=1, charge=0, description=format(r))
            molecule.load()
            return molecule

    uccsd_ansatz = CCSD_Ansatz()
    parameters = uccsd_ansatz.operator_parameters
    qubits = uccsd_ansatz.qubits
    # operators = uccsd_ansatz.operations(uccsd_ansatz.qubits)
    circuit = uccsd_ansatz.circuit
    print(circuit)

    # Get variational study
    result = CPU.get_optimized_state(w=uccsd_ansatz, max_iter=10)
    print(f'Operator expectation value: {result.optimal_value}\nOperator parameters: {result.optimal_parameters}')

    parameters.update(r=result)
    resolved_circuit = QPU.get_resolved_circuit(circuit, parameters)
    print(resolved_circuit)

    # --------------------------

    # Get variational study
    # for i in range(5):
    #     result = CPU.get_optimized_state(w=ansatz, max_iter=10)
    #     print(f'Operator expectation value: {result.optimal_value}\nOperator parameters: {result.optimal_parameters}')
    #
    #     for j, key in enumerate(ansatz.molecule_parameters):
    #         ansatz.molecule_parameters.dict[key] += .5
    #     ansatz.update_molecule(ansatz.molecule_parameters)
    #
    # parameters.update(r=result)
    # resolved_circuit = QPU.get_resolved_circuit(circuit, parameters)
    # print(resolved_circuit)

    # Get molecule study
    # minimization = CPU.get_optimized_ground_state(w=ansatz, qpu_iter=10, cpu_iter=10)
    # print(minimization)
