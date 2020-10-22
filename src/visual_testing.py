# https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec
import numpy as np
import pylatexenc
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix
from qiskit.visualization import plot_state_city, circuit_drawer, plot_state_hinton
from tqdm import tqdm  # For displaying for-loop process to console
import cirq
from openfermion import get_sparse_operator
import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure, Subplot
from typing import List, Tuple, Union, Callable, Dict, Any
from src.error_mitigation import BasisUnitarySet, IBasisGateSingle, reconstruct_from_basis, MAX_MULTI_PROCESSING_COUNT
from src.calculate_minimisation import read_object, write_object
from src.data_containers.helper_interfaces.i_noise_wrapper import INoiseModel, INoiseWrapper
from src.data_containers.helper_interfaces.i_wave_function import IWaveFunction
from src.data_containers.helper_interfaces.i_collection import IHistogramCollection, ISimilarityCollection
from src.circuit_noise_extension import Noisify
from src.data_containers.model_hydrogen import HydrogenAnsatz
from src.processors.processor_quantum import QPU
from src.processors.processor_classic import CPU
from src.error_mitigation import SingleQubitPauliChannel, TwoQubitPauliChannel, IErrorMitigationManager
from QP_QEM_lib import swap_circuit
from src.main import get_log_experiment
from multiprocessing import Pool


def get_density_matrix_plot(density_matrix: np.ndarray, title: str, **kwargs):
    """Returns plt object"""
    plot_state_city(density_matrix, color=['midnightblue', 'red'], title=title, **kwargs)


def get_density_matrix_hinton_plot(density_matrix: np.ndarray, title: str, **kwargs):
    """Returns plt object"""
    plot_state_hinton(density_matrix, title=title, **kwargs)


def setup_density_circuit_plot(density_matrix: np.ndarray, circuit: Union[QuantumCircuit, None], title: str):
    fig = plt.figure()
    proj = None  # '3d'  #

    if circuit is None:
        ax1 = fig.add_subplot(1, 2, 1, projection=proj)
        ax2 = fig.add_subplot(1, 2, 2, projection=proj)
    else:
        ax1 = fig.add_subplot(2, 2, 1, projection=proj)
        ax2 = fig.add_subplot(2, 2, 2, projection=proj)
        ax3 = fig.add_subplot(2, 1, 2)
        # Visualize simple circuit
        circuit.draw(output='mpl', ax=ax3)

    # Visualize density matrix
    get_density_matrix_hinton_plot(density_matrix, title, ax_real=ax1, ax_imag=ax2)

    if proj is not None:
        minmax = (0, 1)
        ax1.set_zlim(minmax)
        ax2.set_zlim(minmax)
    fig.set_size_inches(11, 8)


def setup_before_after_density_plot(density_matrix: np.ndarray, noise_matrix: np.ndarray, circuit: Union[QuantumCircuit, None], title: str):
    fig = plt.figure()
    proj = None  # '3d'  #

    if circuit is None:
        ax1 = fig.add_subplot(1, 2, 1, projection=proj)
        ax2 = fig.add_subplot(1, 2, 2, projection=proj)
    else:
        ax1 = fig.add_subplot(2, 2, 1, projection=proj)
        ax2 = fig.add_subplot(2, 2, 2, projection=proj)
        ax3 = fig.add_subplot(2, 1, 2)
        # Visualize simple circuit
        circuit.draw(output='mpl', ax=ax3)

    # Visualize density matrix
    get_density_matrix_hinton_plot(density_matrix, title, ax_real=ax1)
    get_density_matrix_hinton_plot(noise_matrix, title, ax_real=ax2)

    if proj is not None:
        minmax = (0, 1)
        ax1.set_zlim(minmax)
        ax2.set_zlim(minmax)
    fig.set_size_inches(11, 8)


def setup_before_after_2D_3D_plot(density_matrix: np.ndarray, noise_matrix: np.ndarray, circuit: Union[QuantumCircuit, None], title: str):
    fig = plt.figure()
    proj = '3d'  #

    ax1 = fig.add_subplot(2, 2, 1, projection=None)
    ax2 = fig.add_subplot(2, 2, 2, projection=None)
    ax3 = fig.add_subplot(2, 2, 3, projection=proj)
    ax4 = fig.add_subplot(2, 2, 4, projection=proj)

    # Visualize density matrix
    get_density_matrix_hinton_plot(density_matrix, title, ax_real=ax1)
    get_density_matrix_hinton_plot(noise_matrix, title, ax_real=ax2)
    get_density_matrix_plot(density_matrix, title, ax_real=ax3)
    get_density_matrix_plot(noise_matrix, title, ax_real=ax4)

    minmax = (0, 1)
    ax3.set_zlim(minmax)
    ax4.set_zlim(minmax)

    fig.set_size_inches(11, 8)


def simulate_density_matrix(circuit: cirq.Circuit) -> np.ndarray:
    simulator = cirq.DensityMatrixSimulator(ignore_measurement_results=True)  # Mixed state simulator
    simulated_result = simulator.simulate(program=circuit)  # Include final density matrix
    return simulated_result.final_density_matrix


def single_qubit_identity_circuit():
    dummy_qubits = cirq.LineQubit.range(1)
    # Visualize simple circuit
    circuit = cirq.Circuit(cirq.I.on(dummy_qubits[0]))
    matrix = simulate_density_matrix(circuit)
    # Build a quantum circuit
    qc = QuantumCircuit(1)
    qc.i(0)
    # Plot
    setup_density_circuit_plot(density_matrix=matrix, circuit=qc, title="Example Density Matrix")


def single_qubit_hadamard_circuit():
    dummy_qubits = cirq.LineQubit.range(1)
    # Visualize simple circuit
    circuit = cirq.Circuit(cirq.H.on(dummy_qubits[0]))
    matrix = simulate_density_matrix(circuit)
    # Build a quantum circuit
    qc = QuantumCircuit(1)
    qc.h(0)
    # Plot
    setup_density_circuit_plot(density_matrix=matrix, circuit=qc, title="Example Density Matrix")


def two_qubit_H_CNOT_H():
    dummy_qubits = cirq.LineQubit.range(2)
    # Visualize two qubit circuit
    circuit_02 = cirq.Circuit(cirq.H.on(dummy_qubits[0]))
    circuit_02.append(cirq.CNOT(dummy_qubits[0], dummy_qubits[1]))
    circuit_02.append(cirq.H.on(dummy_qubits[0]))
    matrix = simulate_density_matrix(circuit_02)
    # Build a quantum circuit
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.h(0)
    # Plot
    setup_density_circuit_plot(density_matrix=matrix, circuit=qc, title="Two qubit final density matrix")


def single_qubit_depolar_noise():
    dummy_qubits = cirq.LineQubit.range(1)
    p = .5
    # Visualize simple circuit
    # Clean
    circuit = cirq.Circuit(cirq.H.on(dummy_qubits[0]))
    matrix = simulate_density_matrix(circuit)
    # Noisy
    noise_model = INoiseModel(noise_gates_1q=[cirq.depolarize(p=p)], noise_gates_2q=[], description=f'Depolarize (p={p})')
    circuit_noise = Noisify.introduce_noise(circuit, noise_model.get_callable())
    matrix_noise = simulate_density_matrix(circuit_noise)
    # Build a quantum circuit
    qc = QuantumCircuit(1)
    qc.h(0)
    # Plot
    setup_before_after_2D_3D_plot(density_matrix=matrix, noise_matrix=matrix_noise, circuit=qc, title=f'Effect of {noise_model.get_description()} on the Density matrix')


# Depricated
def single_qubit_depolar_noise2():
    dummy_qubits = cirq.LineQubit.range(1)
    p = .5
    # Visualize simple circuit

    noise_model = INoiseModel(noise_gates_1q=[cirq.depolarize(p=p)], noise_gates_2q=[], description=f'Depolarize (p={p})')

    effective_unitary = np.array([[1, 0], [0, 0]])
    effective_unitary = cirq.unitary(cirq.H) @ (effective_unitary @ cirq.unitary(cirq.H).conj().transpose())
    effective_unitary = noise_model.get_effective_gate(effective_unitary)
    noisy_gate = IBasisGateSingle(unitary=effective_unitary)

    circuit = cirq.Circuit(noisy_gate.on(dummy_qubits[0]))
    circuit = Noisify.introduce_noise(circuit, noise_model.get_callable())
    matrix = simulate_density_matrix(circuit)
    # Build a quantum circuit
    qc = QuantumCircuit(1)
    qc.h(0)
    # Plot
    setup_density_circuit_plot(density_matrix=effective_unitary, circuit=qc, title=f'Operations followed by {noise_model.get_description()}')


def single_qubit_dephase_noise():
    dummy_qubits = cirq.LineQubit.range(1)
    p = 0.5
    # Visualize simple circuit

    # circuit = cirq.Circuit(cirq.H.on(dummy_qubits[0]))  # [cirq.H.on(dummy_qubits[0]), cirq.Y.on(dummy_qubits[0])]
    # noise_model = INoiseModel(noise_gates=[cirq.phase_damp(gamma=p)], description=f'Phase damping (p={p})')
    # circuit = Noisify.introduce_noise(circuit, noise_model.get_callable())
    # matrix = simulate_density_matrix(circuit)
    # # Build a quantum circuit
    # qc = QuantumCircuit(1)
    # qc.h(0)
    # # Plot
    # setup_density_circuit_plot(density_matrix=matrix, circuit=qc, title=f'Operations followed by {noise_model.get_description()}')

    # Clean
    circuit = cirq.Circuit(cirq.H.on(dummy_qubits[0]))
    matrix = simulate_density_matrix(circuit)
    # Noisy
    noise_model = INoiseModel(noise_gates_1q=[cirq.phase_damp(gamma=p)], noise_gates_2q=[], description=f'Phase damping (p={p})')
    circuit_noise = Noisify.introduce_noise(circuit, noise_model.get_callable())
    matrix_noise = simulate_density_matrix(circuit_noise)
    # Build a quantum circuit
    qc = QuantumCircuit(1)
    qc.h(0)
    # Plot
    setup_before_after_2D_3D_plot(density_matrix=matrix, noise_matrix=matrix_noise, circuit=qc, title=f'Effect of {noise_model.get_description()} on the Density matrix')


def single_qubit_ampdamp_noise():
    dummy_qubits = cirq.LineQubit.range(1)
    p = 0.5
    # Visualize simple circuit

    # circuit = cirq.Circuit(cirq.H.on(dummy_qubits[0]))  # [cirq.H.on(dummy_qubits[0]), cirq.Y.on(dummy_qubits[0])]
    # noise_model = INoiseModel(noise_gates=[cirq.amplitude_damp(gamma=p)], description=f'Amplitude damping (p={p})')
    # circuit = Noisify.introduce_noise(circuit, noise_model.get_callable())
    # matrix = simulate_density_matrix(circuit)
    # # Build a quantum circuit
    # qc = QuantumCircuit(1)
    # qc.h(0)
    # # Plot
    # setup_density_circuit_plot(density_matrix=matrix, circuit=qc, title=f'Operations followed by {noise_model.get_description()}')

    # Clean
    circuit = cirq.Circuit(cirq.H.on(dummy_qubits[0]))
    matrix = simulate_density_matrix(circuit)
    # Noisy
    noise_model = INoiseModel(noise_gates_1q=[cirq.amplitude_damp(gamma=p)], noise_gates_2q=[], description=f'Amplitude damping (p={p})')
    circuit_noise = Noisify.introduce_noise(circuit, noise_model.get_callable())
    matrix_noise = simulate_density_matrix(circuit_noise)
    # Build a quantum circuit
    qc = QuantumCircuit(1)
    qc.h(0)
    # Plot
    setup_before_after_2D_3D_plot(density_matrix=matrix, noise_matrix=matrix_noise, circuit=qc, title=f'Effect of {noise_model.get_description()} on the Density matrix')


def hydrogen_model_initial_state():
    uccsd_ansatz = HydrogenAnsatz()
    parameters = uccsd_ansatz.operator_parameters
    circuit = QPU.get_initial_state_circuit(uccsd_ansatz)
    circuit.append([cirq.I.on(uccsd_ansatz.qubits[2]), cirq.I.on(uccsd_ansatz.qubits[3])])
    matrix = simulate_density_matrix(circuit)
    # Plot
    setup_density_circuit_plot(density_matrix=matrix, circuit=None, title=f'Initial (Hartree Fock) state for Hydrogen Model')


def hydrogen_model_density_state():
    uccsd_ansatz = HydrogenAnsatz()
    parameters = uccsd_ansatz.operator_parameters
    circuit = QPU.get_initial_state_circuit(uccsd_ansatz)
    circuit.append(uccsd_ansatz.circuit)
    result = CPU.get_optimized_state(w=uccsd_ansatz, max_iter=1000)
    # Resolve circuit
    parameters.update(r=result)
    circuit = QPU.get_resolved_circuit(circuit, parameters)

    matrix = simulate_density_matrix(circuit)
    # Plot
    setup_density_circuit_plot(density_matrix=matrix, circuit=None, title=f'Hydrogen Model (r = 0.7414) density state after evolution')


def hydrogen_model_transition_state():
    # Visualize simple circuit
    uccsd_ansatz = HydrogenAnsatz()
    parameters = uccsd_ansatz.operator_parameters
    circuit = QPU.get_initial_state_circuit(uccsd_ansatz)
    circuit.append([cirq.I.on(uccsd_ansatz.qubits[2]), cirq.I.on(uccsd_ansatz.qubits[3])])

    # Before
    matrix_before = simulate_density_matrix(circuit)
    # After
    circuit.append(uccsd_ansatz.circuit)
    result = CPU.get_optimized_state(w=uccsd_ansatz, max_iter=1000)
    # Resolve circuit
    parameters.update(r=result)
    circuit = QPU.get_resolved_circuit(circuit, parameters)
    matrix_after = simulate_density_matrix(circuit)
    # Plot
    setup_before_after_density_plot(density_matrix=matrix_before, noise_matrix=matrix_after, circuit=None,
                                    title=f'Hydrogen Model (r = 0.7414) density state before and after evolution')


def gst_hadamard_ideal():
    effective_unitary = np.array([[1, 0], [0, 0]])
    effective_unitary = cirq.unitary(cirq.H) @ (effective_unitary @ cirq.unitary(cirq.H).conj().transpose())  # rho under Hadamard

    matrix = BasisUnitarySet.pauli_transfer_matrix(gate=effective_unitary)[1]
    # Plot
    setup_before_after_2D_3D_plot(density_matrix=effective_unitary, noise_matrix=matrix, circuit=None, title=r'Gate set tomography of Density Matrix')

    print(effective_unitary)
    print(matrix)

    qp, basis = BasisUnitarySet.get_decomposition(gate=effective_unitary)
    goal = reconstruct_from_basis(qp, basis)

    print(goal)


def gst_identity_ideal():
    effective_unitary = np.array([[1, 0], [0, 0]])
    effective_unitary = cirq.unitary(cirq.I) @ (effective_unitary @ cirq.unitary(cirq.I).conj().transpose())  # rho under Hadamard

    matrix = BasisUnitarySet.pauli_transfer_matrix(gate=effective_unitary)[1]
    # Plot
    setup_before_after_2D_3D_plot(density_matrix=effective_unitary, noise_matrix=matrix, circuit=None, title=r'Gate set tomography of Density Matrix')

    print(effective_unitary)
    print(matrix)

    qp, basis = BasisUnitarySet.get_decomposition(gate=effective_unitary)
    goal = reconstruct_from_basis(qp, basis)

    print(qp)


def sampling_noise_scaling():
    """
    Experiment: Sampling noise scaling.
    :return:
    """
    # Plotting parameters

    # Data
    ansatz = HydrogenAnsatz()
    parameters = ansatz.operator_parameters
    p = 1e-4

    # Construct noise model
    channel_1q = [SingleQubitPauliChannel(p_x=p, p_y=p, p_z=6 * p)]
    channel_2q = [TwoQubitPauliChannel(p_x=p, p_y=p, p_z=6 * p)]
    noise_model = INoiseModel(noise_gates_1q=channel_1q, noise_gates_2q=channel_2q,
                              description=f'asymmetric depolarization (p_tot={16 * p})')
    noisy_ansatz = INoiseWrapper(ansatz, noise_model)

    # Construct circuit
    circuit = noisy_ansatz.get_clean_circuit()
    # Get variational study
    result = CPU.get_optimized_state(w=ansatz, max_iter=1000)
    parameters.update(r=result)
    resolved_circuit = QPU.get_resolved_circuit(circuit, parameters)

    # Get Hamiltonian objective
    qubit_operator = QPU.get_hamiltonian_evaluation_operator(ansatz)
    H_observable = get_sparse_operator(qubit_operator)  # Observable

    def experiment_func(process_circuit_count: int) -> float:
        return CPU.get_mitigated_expectation(clean_circuit=resolved_circuit, noise_model=noise_model,
                                             process_circuit_count=process_circuit_count,
                                             hamiltonian_objective=H_observable)

    shot_list = [int(shot) for shot in np.logspace(1, 4, 10)]
    experiments = get_log_experiment(shot_list=shot_list, expectation=result.optimal_value, experiment=experiment_func, reps=10)
    # Values
    y = np.array(list(experiments.values()))
    x = np.log10(np.array(shot_list))
    # Fit
    try:
        curve_fit = np.polyfit(x, y, 1)
        y_data = curve_fit[0] * x + curve_fit[1]
        plt.plot(x, y_data, '--', label='Linear fit log(y) = {}log(x) + {}'.format(curve_fit[0].round(4), curve_fit[1].round(4)))
    except np.linalg.LinAlgError:
        pass

    plt.plot(x, np.array(list(experiments.values())), 'o', label='Average error')
    plt.title('H2 ansatz circuit with asymmetric depolarizing noise and error mitigation')
    plt.xlabel('log10(sampling circuit count)')
    plt.ylabel('Error in H2 ground state estimation')

    plt.legend(frameon=False)


class ObservableMeasureClass:
    def __init__(self, w: IWaveFunction, c: cirq.Circuit, n_op: Callable[[Tuple[cirq.Qid]], List[cirq.Operation]], m: int):
        self._w = w
        self._c = c
        self._n_op = n_op
        self._m = m

    def exp_measure(self, index: int):
        func, cost = self._w.observable_measurement()
        return func(self._c, self._n_op, self._m)


def hamiltonian_density_vs_measure():
    """Plots the H2 expectation value at ground state for different noise models"""
    # Hydrogen ansatz
    ansatz = HydrogenAnsatz()
    # Get optimization
    result = CPU.get_optimized_state(w=ansatz, max_iter=1000)
    parameters = ansatz.operator_parameters
    parameters.update(r=result)
    # Get Measurement function
    observable_process, circuit_cost = ansatz.observable_measurement()

    def get_noise_model(_p: float) -> INoiseModel:
        channel_1q = [SingleQubitPauliChannel(p_x=_p, p_y=_p, p_z=6 * _p)]
        channel_2q = [TwoQubitPauliChannel(p_x=_p, p_y=_p, p_z=6 * _p)]
        return INoiseModel(noise_gates_1q=channel_1q, noise_gates_2q=channel_2q, description=f'asymmetric depolarization (p_tot={16 * _p})')

    prob_list = [float(1/shot) for shot in np.logspace(1, 4, 4)]
    meas_reps = [int(shot) for shot in np.logspace(0, 4, 10)]
    count = 50

    error_list = np.ndarray(shape=(len(prob_list), len(meas_reps)))
    error_var_list = np.ndarray(shape=(len(prob_list), len(meas_reps)))
    fidelity_list = []
    measure_list = []
    print(f'Start measuring. For {len(prob_list)} probabilities')
    for i, prob in enumerate(prob_list):
        noise_model = get_noise_model(_p=prob)
        noisy_ansatz = INoiseWrapper(ansatz, noise_model)
        circuit = noisy_ansatz.get_noisy_circuit()
        # Density matrix approach
        exp_fidelity = QPU.get_simulated_noisy_expectation_value(w=noisy_ansatz, r_c=circuit, r=parameters.get_resolved())
        # fidelity_list.append(exp_fidelity)
        for j, meas in enumerate(tqdm(meas_reps)):
            # Measurement approach
            resolved_circuit = QPU.get_resolved_circuit(circuit, parameters)
            process_measure = ObservableMeasureClass(ansatz, resolved_circuit, noise_model.get_operators, meas)
            with Pool(MAX_MULTI_PROCESSING_COUNT) as p:
                exp_measure = p.map(process_measure.exp_measure, range(count))
            # exp_measure = np.array([observable_process(resolved_circuit, noise_model.get_operators, meas) for i in range(count)])
            # measure_list.append(exp_measure)

            error_list[i][j] = abs(exp_fidelity - np.mean(exp_measure))
            error_var_list[i][j] = abs(np.std(exp_measure))
    print('Measuring complete')

    # fidelity_n_list = []
    # measure_n_list = []
    # for meas in tqdm(meas_reps):
    #     noise_model = get_noise_model(_p=p)
    #     noisy_ansatz = INoiseWrapper(ansatz, noise_model)
    #     circuit = noisy_ansatz.get_noisy_circuit()
    #     # Density matrix approach
    #     exp_fidelity = QPU.get_simulated_noisy_expectation_value(w=noisy_ansatz, r_c=circuit, r=parameters.get_resolved())
    #     fidelity_n_list.append(exp_fidelity)
    #     # Measurement approach
    #     resolved_circuit = QPU.get_resolved_circuit(circuit, parameters)
    #     exp_measure = np.array([observable_process(resolved_circuit, noise_model.get_operators, meas) for i in range(count)])
    #     measure_n_list.append(exp_measure)

    # Values
    # x = np.log10(np.array(meas_reps))
    # y = [np.mean(measure) for measure in measure_list]
    # y_var = [np.std(measure) for measure in measure_list]
    # y_n = [np.mean(measure) for measure in measure_n_list]
    # y_var_n = [np.std(measure) for measure in measure_n_list]

    # min_diff = abs(y[-1] - fidelity_list[-1])
    # min_diff_n = abs(y_n[-1] - fidelity_n_list[-1])

    fig = plt.figure()
    ax = plt.axes()
    ax.set_xscale("log")
    ax.set_yscale("log")
    for k, prob_meas in enumerate(error_list):
        # ax.errorbar(meas_reps, prob_meas, yerr=error_var_list[k], fmt='b', label=f'Difference Error (p={prob_list[k]})')
        ax.loglog(meas_reps, prob_meas, label=f'Difference Error (p={prob_list[k]})')

    # plt.plot(x, fidelity_list, 'o', label='Density matrix approach')
    # plt.errorbar(x, y, yerr=y_var, linestyle='-', marker='^', label=f'Measurement approach')  #  e={round(min_diff,6)}
    # plt.plot(x, fidelity_n_list, 'o', label='Density matrix approach (noisy)')
    # plt.errorbar(x, y_n, yerr=y_var_n, linestyle='-', marker='^', label=f'Measurement approach (noisy)')  #  e={round(min_diff_n,6)}
    plt.title(f'H2 ansatz circuit comparing density matrix vs measurement observable (Asymmetric Pauli noise)')
    plt.xlabel('Measurement repetition')
    plt.ylabel('Error in expectation value at ground state')
    # Display grid
    plt.grid(True, which="both")

    plt.legend(frameon=True)


def similarity_plot_swap_circuit(overwrite: bool, prob: float = 1e-4, measure_count: int = 100, identifier_count: int = 1000):
    """Performs a multitude of error mitigation protocols at a certain noise probability.
    Plots histogram of results versus the respective noisy simulations."""
    # Construct noise wrapper
    channel_1q = [SingleQubitPauliChannel(p_x=prob, p_y=prob, p_z=6*prob)]  # , SingleQubitLeakageChannel(p=8 * prob)]
    channel_2q = [TwoQubitPauliChannel(p_x=prob, p_y=prob, p_z=6*prob)]  # , TwoQubitLeakageChannel(p=8 * prob)]
    noise_model = INoiseModel(noise_gates_1q=channel_1q, noise_gates_2q=channel_2q, description=f'asymmetric depolarization (p_tot={16 * prob})')

    if overwrite:
        # Settings
        # prob = 1e-4
        # measure_count = 100
        # identifier_count = 1000
        n = 3
        # Construct circuit
        circuit = swap_circuit(n, cirq.LineQubit.range(2 * n + 1), True)
        # Error Mitigation Manager
        manager = IErrorMitigationManager(clean_circuit=circuit, noise_model=INoiseModel.empty(), hamiltonian_objective=None)
        manager.set_identifier_range(identifier_count)
        mu_ideal = manager.get_mu_effective(error_mitigation=False, density_representation=True, meas_reps=1)[1]
        manager = IErrorMitigationManager(clean_circuit=circuit, noise_model=noise_model, hamiltonian_objective=None)

        result_noisy = []
        result_mitigated = []
        for i in tqdm(range(measure_count), desc='Processing error mitigation functions'):
            manager.set_identifier_range(identifier_count)
            result_noisy.append(manager.get_mu_effective(error_mitigation=False, density_representation=False, meas_reps=1)[1])
            result_mitigated.append(manager.get_mu_effective(error_mitigation=True, density_representation=False, meas_reps=1)[1])

        # Store Collection
        settings = ISimilarityCollection.get_settings(probability=prob, measure_count=measure_count, identifier_count=identifier_count)
        data = ISimilarityCollection(data_noisy=result_noisy, data_mitigated=result_mitigated, mu_ideal=mu_ideal, settings=settings)
        write_object(data, f'SWAP_Similarity_{settings}')
    else:
        settings = ISimilarityCollection.get_settings(probability=prob, measure_count=measure_count, identifier_count=identifier_count)
        data = read_object(f'SWAP_Similarity_{settings}')

    # Plotting Histogram
    n_bins = 20
    fig, axs = plt.subplots(1, 1, tight_layout=True)
    circuit_name = f'Swap circuit using measurements'
    fig.suptitle(f'{circuit_name} (#mitigation circuits={identifier_count})', fontsize=16)
    plot_title = f'(Info: {noise_model.get_description()})'
    axs.title.set_text(plot_title)
    axs.set_xlabel(f'Expectation value after {identifier_count} experiments')  # [{x_lim[0]}, {x_lim[1]}] X axis label
    axs.set_ylabel(f'Frequency of obtaining this result')  # Y axis label

    # Data
    mu_noisy = data.data_noisy.get_mean  # np.mean(result_noisy)
    mu_mitigated = data.data_mitigated.get_mean  # np.mean(result_mitigated)
    std_mitigated = data.data_mitigated.get_std
    mu_ideal = data.mu_ideal
    axs.hist(data.data_noisy.get_data, bins=n_bins, edgecolor='black', alpha=0.7, color='#fc9003', label=f'No error mitigation')
    axs.hist(data.data_mitigated.get_data, bins=n_bins, edgecolor='black', alpha=0.7, color='#1ee300', label=f'Quasiprobability')
    axs.axvline(x=mu_ideal, linewidth=1, color='r', label=r'$\mu_{ideal}$ = ' + f'{np.round(mu_ideal, 5)}')
    axs.axvline(x=mu_noisy, linewidth=1, color='#fc9003', label=r'$E[\mu_{noisy}$] = ' + f'{np.round(mu_noisy, 5)}')
    axs.axvline(x=mu_mitigated, linewidth=1, color='#1ee300', label=r'E[$\mu_{mitigated}$] = ' + f'{np.round(mu_mitigated, 5)}\n' + r'|$\epsilon$| = ' + f'{np.round(abs(mu_mitigated - mu_ideal), 5)}\n' + r'$\sigma$ = ' + f'{np.round(std_mitigated, 5)}')
    axs.legend()


def similarity_plot_hydrogen_circuit(overwrite: bool, prob: float = 1e-4, measure_count: int = 100, identifier_count: int = 1000):
    """Performs a multitude of error mitigation protocols at a certain noise probability.
    Plots histogram of results versus the respective noisy simulations."""
    # Construct noise wrapper
    channel_1q = [SingleQubitPauliChannel(p_x=prob, p_y=prob, p_z=6 * prob)]  # , SingleQubitLeakageChannel(p=8 * prob)]
    channel_2q = [TwoQubitPauliChannel(p_x=prob, p_y=prob, p_z=6 * prob)]  # , TwoQubitLeakageChannel(p=8 * prob)]
    noise_model = INoiseModel(noise_gates_1q=channel_1q, noise_gates_2q=channel_2q, description=f'asymmetric depolarization (p_tot={16 * prob})')

    if overwrite:
        # Settings
        # prob = 1e-4
        # measure_count = 100
        # identifier_count = 1000
        # Get optimization
        # result = CPU.get_optimized_state(w=ansatz, max_iter=1000)

        # Hydrogen ansatz
        ansatz = HydrogenAnsatz()
        noisy_parameters = ansatz.operator_parameters
        noisy_ansatz = INoiseWrapper(w_class=ansatz, noise_model=noise_model)

        # Construct circuit
        circuit = noisy_ansatz.get_clean_circuit()

        # Ideal manager
        # manager = IErrorMitigationManager(clean_circuit=circuit_ideal_param, noise_model=INoiseModel.empty(), hamiltonian_objective=ansatz)
        # manager.set_identifier_range(identifier_count)
        # mu_ideal = manager.get_mu_effective(error_mitigation=False, density_representation=True, meas_reps=1)[1]
        mu_ideal = -1.13727  # Hardcoded

        result_noisy = []
        result_mitigated = []
        for i in tqdm(range(measure_count), desc='Processing error mitigation functions'):
            # Calculate Ideal and noisy circuit parameters
            # opt_result = CPU.get_optimized_state(w=ansatz, max_iter=1000)
            # Realistically calculate ideal circuit parameters under noisy environment
            exp_value, opt_params = CPU.get_custom_optimized_state(n_w=noisy_ansatz, max_cpu_iter=10, max_qpu_iter=10000)
            noisy_parameters.update(v=opt_params)

            # circuit_ideal_param = QPU.get_resolved_circuit(c=circuit, p=opt_result.optimized_parameter)
            circuit_noisy_param = QPU.get_resolved_circuit(c=circuit, p=noisy_parameters)

            # Error Mitigation Manager
            # Noisy manager
            manager = IErrorMitigationManager(clean_circuit=circuit_noisy_param, noise_model=noise_model, hamiltonian_objective=ansatz)
            manager.set_identifier_range(identifier_count)
            result_noisy.append(manager.get_mu_effective(error_mitigation=False, density_representation=False, meas_reps=1)[1])
            result_mitigated.append(manager.get_mu_effective(error_mitigation=True, density_representation=False, meas_reps=1)[1])

        # Store Collection
        settings = ISimilarityCollection.get_settings(probability=prob, measure_count=measure_count, identifier_count=identifier_count)
        data = ISimilarityCollection(data_noisy=result_noisy, data_mitigated=result_mitigated, mu_ideal=mu_ideal, settings=settings)
        write_object(data, f'H2_Similarity_{settings}')
    else:
        settings = ISimilarityCollection.get_settings(probability=prob, measure_count=measure_count, identifier_count=identifier_count)
        data = read_object(f'H2_Similarity_{settings}')

    # Plotting Histogram
    n_bins = 20
    fig, axs = plt.subplots(1, 1, tight_layout=True)
    circuit_name = f'H2 ansatz circuit using realistic measurements'
    fig.suptitle(f'{circuit_name} (#mitigation circuits={identifier_count})', fontsize=16)
    plot_title = f'(Info: {noise_model.get_description()})'
    axs.title.set_text(plot_title)
    axs.set_xlabel(f'Expectation value after {identifier_count} experiments')  # [{x_lim[0]}, {x_lim[1]}] X axis label
    axs.set_ylabel(f'Frequency of obtaining this result')  # Y axis label

    # Data
    mu_noisy = data.data_noisy.get_mean  # np.mean(result_noisy)
    mu_mitigated = data.data_mitigated.get_mean  #  np.mean(result_mitigated)
    std_mitigated = data.data_mitigated.get_std
    mu_ideal = data.mu_ideal
    axs.hist(data.data_noisy.get_data, bins=n_bins, edgecolor='black', alpha=0.7, color='#fc9003', label=f'No error mitigation')
    axs.hist(data.data_mitigated.get_data, bins=n_bins, edgecolor='black', alpha=0.7, color='#1ee300', label=f'Quasiprobability')
    axs.axvline(x=mu_ideal, linewidth=1, color='r', label=r'$\mu_{ideal}$ = ' + f'{np.round(mu_ideal, 5)}')
    axs.axvline(x=mu_noisy, linewidth=1, color='#fc9003', label=r'$E[\mu_{noisy}$] = ' + f'{np.round(mu_noisy, 5)}')
    axs.axvline(x=mu_mitigated, linewidth=1, color='#1ee300', label=r'E[$\mu_{mitigated}$] = ' + f'{np.round(mu_mitigated, 5)}\n' + r'|$\epsilon$| = ' + f'{np.round(abs(mu_mitigated - mu_ideal), 5)}\n' + r'$\sigma$ = ' + f'{np.round(std_mitigated, 5)}')
    axs.legend()


class ParameterOptimizerClass:
    def __init__(self, n_w: INoiseWrapper):
        self._n_w = n_w

    def optimize(self, index: int):
        return CPU.get_custom_optimized_state(n_w=self._n_w, max_cpu_iter=10, max_qpu_iter=10000)


def circuit_parameter_optimization(overwrite: bool):
    # Hydrogen ansatz
    ansatz = HydrogenAnsatz()
    # Get optimization
    count = 20
    mu_ideal = -1.13727  # Hardcoded
    # mu_ideal, opt_ideal = ParameterOptimizerClass(INoiseWrapper(ansatz, INoiseModel.empty())).optimize(0)
    opt_ideal = 0.015625  # Hardcoded opt_ideal[0]

    if overwrite:
        exp_list = []
        opt_list = []
        # Optimize circuit using realistic measurements
        # prob_list = [0, 1e-4, 5e-4, 1e-3, 5e-3]
        prob_list = [1 / shot for shot in np.logspace(2, 5, 14)]
        # prob_list.append(0)
        prob_list = np.flip(np.array(prob_list))
        for prob in prob_list:
            # Construct noise wrapper
            channel_1q = [SingleQubitPauliChannel(p_x=prob, p_y=prob, p_z=6 * prob)]
            channel_2q = [TwoQubitPauliChannel(p_x=prob, p_y=prob, p_z=6 * prob)]
            noise_model = INoiseModel(noise_gates_1q=channel_1q, noise_gates_2q=channel_2q, description=f'asymmetric depolarization (p_tot={16 * prob})')
            noisy_ansatz = INoiseWrapper(w_class=ansatz, noise_model=noise_model)

            process_measure = ParameterOptimizerClass(n_w=noisy_ansatz)
            # print(f'Start calculation')
            with Pool(MAX_MULTI_PROCESSING_COUNT) as p:
                results = p.map(process_measure.optimize, tqdm(range(count), desc=f'Optimizing circuits'))

            new_exp_list = []
            new_opt_list = []
            for value, params in results:
                new_exp_list.append(value)
                new_opt_list.append(params[0])
            exp_list.append(new_exp_list)
            opt_list.append(new_opt_list)
        # Store data
        data = (exp_list, opt_list, prob_list)
        write_object(data, f'H2_Optimizing_Parameters_02')
    else:
        data = read_object(f'H2_Optimizing_Parameters_02')
    exp_list, opt_list, prob_list = data

    fig, axs = plt.subplots(1, 1, tight_layout=True)
    # plot_title = f'Optimized circuit parameters using realistic measurements'
    # axs.title.set_text(plot_title)
    # axs.set_xlabel(f'Expectation value after optimization')  # X axis label
    # axs.set_ylabel(f'Optimized circuit parameter value '+r'[$\frac{1}{2} \pi$]')  # Y axis label
    # # Rescale to multiples of pi
    # opt_list = np.array(opt_list) / (0.5 * np.pi)  # Parameters have a range of [0, pi]
    # axs.set_ylim([-1, 1])
    # for i in range(len(exp_list)):
    #     axs.plot(exp_list[i], opt_list[i], '.', label=f'Optimized parameters (noise p={prob_list[i]})')
    # axs.axvline(x=mu_ideal, linewidth=1, color='r', label=r'$\mu_{ideal}$ = ' + f'{np.round(mu_ideal, 5)}')

    plot_title = f'Optimized circuit parameters using realistic measurements'
    axs.title.set_text(plot_title)
    axs.set_xlabel(f'Noise magnitude (Asymmetric Pauli noise)')  # X axis label
    # axs.set_ylabel(f'Optimized circuit parameter value '+r'[$\frac{1}{2} \pi$]')  # Y axis label
    axs.set_ylabel(f'Absolute difference error compared to noiseless parameter: '+r'|$\epsilon$|=|$par - par_{ideal}$| [$\frac{1}{2} \pi$]')  # Y axis label

    # Rescale to multiples of pi
    opt_list = np.array(opt_list) / (0.5 * np.pi)  # Parameters have a range of [0, pi]
    # Set log scale difference
    vfunc = np.vectorize(lambda x: abs(x - opt_ideal))
    opt_list = vfunc(opt_list)
    # Mean
    mean_opt_list = [np.mean(np.array(opt)) for opt in opt_list]
    # Standard deviation
    std_opt_list = [np.std(np.array(opt)) for opt in opt_list]

    # axs.set_ylim([-1, 1])
    for i in range(len(exp_list)):
        x = [prob_list[i] for j in exp_list[i]]
        axs.plot(x, opt_list[i], '.')  # , label=f'Eigenvalue: {np.round(np.mean(np.array(exp_list[i])), 2)}'
    axs.errorbar(prob_list, mean_opt_list, yerr=std_opt_list, fmt='o', alpha=0.5, ecolor='black', capsize=10)  # , label=f'Difference Error (p={prob_list[k]})'
    axs.axhline(y=0, linewidth=1, color='r', label=r'$par_{ideal}$ = ' + f'{np.round(opt_ideal, 5)}')
    axs.set_xscale("log")
    axs.set_yscale("log")

    # Display grid
    plt.grid(True, which="both")
    axs.legend()


class RawAndMitigatedClass:
    def __init__(self, n_w: INoiseWrapper, noise_model: INoiseModel, max_qpu_iter: int):
        self._n_w = n_w
        self._circuit = n_w.get_clean_circuit()
        self._n_model = noise_model
        self._iter = max_qpu_iter

    def process(self):
        # Realistically calculate ideal circuit parameters under noisy environment
        exp_noisy_value, opt_params = CPU.get_custom_optimized_state(n_w=self._n_w, max_cpu_iter=10, max_qpu_iter=self._iter)
        ansatz_parameters = self._n_w.operator_parameters
        ansatz_parameters.update(v=opt_params)

        # circuit_ideal_param = QPU.get_resolved_circuit(c=circuit, p=opt_result.optimized_parameter)
        circuit_noisy_param = QPU.get_resolved_circuit(c=self._circuit, p=ansatz_parameters)

        # Error Mitigation Manager
        # Noisy manager
        manager = IErrorMitigationManager(clean_circuit=circuit_noisy_param, noise_model=self._n_model, hamiltonian_objective=self._n_w)
        manager.set_identifier_range(self._iter)
        exp_mitig_value = manager.get_mu_effective(error_mitigation=True, density_representation=False, meas_reps=1)[1]
        return exp_noisy_value, exp_mitig_value


def flatten(l: List[List[Any]]) -> np.ndarray:
    return np.array([item for sublist in l for item in sublist])


def full_raw_vs_mitigation_per_noise(overwrite: bool, measure_count: int, identifier_count: int):
    # Get optimization
    mu_ideal = -1.13727  # Hardcoded
    if overwrite:
        # Hydrogen ansatz
        ansatz = HydrogenAnsatz()

        exp_noisy_list = []
        exp_mitig_list = []
        # Optimize circuit using realistic measurements
        prob_list = [1 / shot for shot in np.logspace(3, 5, 5)]
        # prob_list.append(0)
        prob_list = np.flip(np.array(prob_list))
        for prob in prob_list:
            # Construct noise wrapper
            channel_1q = [SingleQubitPauliChannel(p_x=prob, p_y=prob, p_z=6 * prob)]
            channel_2q = [TwoQubitPauliChannel(p_x=prob, p_y=prob, p_z=6 * prob)]
            noise_model = INoiseModel(noise_gates_1q=channel_1q, noise_gates_2q=channel_2q, description=f'asymmetric depolarization (p_tot={16 * prob})')
            noisy_ansatz = INoiseWrapper(w_class=ansatz, noise_model=noise_model)

            process_measure = RawAndMitigatedClass(n_w=noisy_ansatz, noise_model=noise_model, max_qpu_iter=identifier_count)
            results = []
            for i in tqdm(range(measure_count), desc='Processing error mitigation functions'):
                results.append(process_measure.process())
            # Results: List[Tuple[noisy exp value, mitigated exp value]]

            new_exp_noisy_list = []
            new_exp_mitig_list = []
            for noisy_exp, mitigated_exp in results:
                new_exp_noisy_list.append(noisy_exp)
                new_exp_mitig_list.append(mitigated_exp)
            exp_noisy_list.append(new_exp_noisy_list)
            exp_mitig_list.append(new_exp_mitig_list)
        # Store data
        data = (exp_noisy_list, exp_mitig_list, prob_list)
        write_object(data, f'H2_Measure_Raw_vs_Mitigated')  # _Detailed
    else:
        data = read_object(f'H2_Measure_Raw_vs_Mitigated')  # _Detailed
    exp_noisy_list, exp_mitig_list, prob_list = data

    fig, axs = plt.subplots(1, 1, tight_layout=True)

    plot_title = f'Energy expectation value using realistic measurements (#mitigation circuits={identifier_count})'  #  depending on noise magnitude
    axs.title.set_text(plot_title)
    axs.set_xlabel(f'Noise magnitude (Asymmetric Pauli noise)')  # X axis label
    # axs.set_ylabel(f'Expectation value after optimization')  # Y axis label
    axs.set_ylabel(f'Absolute difference error compared to noiseless optimization: '+r'|$\epsilon$|=|$\mu - \mu_{ideal}$|')  # Y axis label

    # Set log scale difference
    abs_diff = np.vectorize(lambda x: abs(x - mu_ideal))
    exp_noisy_list = abs_diff(exp_noisy_list)
    exp_mitig_list = abs_diff(exp_mitig_list)

    using_mean = len(exp_noisy_list[0]) > 2

    y_noisy = flatten(exp_noisy_list)
    y_mitig = flatten(exp_mitig_list)

    x_prob = [[prob_list[i]]*len(sublist) for i, sublist in enumerate(exp_noisy_list)]
    x_prob = flatten(x_prob)

    axs.plot(x_prob, y_noisy, '.', color='blue', label=f'Noisy Energy expectation')
    axs.plot(x_prob, y_mitig, '.', color='green', label=f'Mitigated Energy expectation')

    if using_mean:
        y_noisy_mean = [np.mean(sublist) for sublist in exp_noisy_list]
        y_noisy_std = [np.std(sublist) for sublist in exp_noisy_list]
        y_mitig_mean = [np.mean(sublist) for sublist in exp_mitig_list]
        y_mitig_std = [np.std(sublist) for sublist in exp_mitig_list]
        axs.errorbar(prob_list, y_noisy_mean, yerr=y_noisy_std, fmt='o', alpha=0.5, color='blue', ecolor='blue', capsize=10)
        axs.errorbar(prob_list, y_mitig_mean, yerr=y_mitig_std, fmt='o', alpha=0.5, color='green', ecolor='green', capsize=10)

    axs.axhline(y=0, linewidth=1, color='r', label=r'$\mu_{ideal}$ = ' + f'{np.round(mu_ideal, 5)}')
    axs.set_xscale("log")
    axs.set_yscale("log")

    # Display grid
    plt.grid(True, which="both")
    axs.legend()


def testing():
    """Show variety of tests"""
    # single_qubit_identity_circuit()
    # single_qubit_hadamard_circuit()
    # single_qubit_depolar_noise()
    # single_qubit_dephase_noise()
    # single_qubit_ampdamp_noise()

    # two_qubit_H_CNOT_H()

    # hydrogen_model_initial_state()
    # hydrogen_model_density_state()
    # hydrogen_model_transition_state()

    # gst_hadamard_ideal()
    # gst_identity_ideal()

    # sampling_noise_scaling()
    # hamiltonian_density_vs_measure()

    # Build data
    master_overwrite = False
    similarity_plot_swap_circuit(overwrite=master_overwrite, prob=1e-4, measure_count=100, identifier_count=10000)
    similarity_plot_hydrogen_circuit(overwrite=master_overwrite, prob=1e-4, measure_count=100, identifier_count=10000)
    # similarity_plot_swap_circuit(overwrite=master_overwrite, prob=1e-3, measure_count=100, identifier_count=10000)
    # similarity_plot_hydrogen_circuit(overwrite=master_overwrite, prob=1e-3, measure_count=100, identifier_count=10000)

    # circuit_parameter_optimization(False)
    # full_raw_vs_mitigation_per_noise(False, 1, 1000)


if __name__ == '__main__':
    testing()
    plt.show()
