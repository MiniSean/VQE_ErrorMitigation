# https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec
import numpy as np
import pylatexenc
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix
from qiskit.visualization import plot_state_city, circuit_drawer, plot_state_hinton
from tqdm import tqdm  # For displaying for-loop process to console
import cirq
import openfermioncirq
import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure, Subplot
from typing import List, Tuple, Union
from src.error_mitigation import BasisUnitarySet, IBasisGateSingle, reconstruct_from_basis
from src.data_containers.helper_interfaces.i_noise_wrapper import INoiseModel, INoiseWrapper
from src.circuit_noise_extension import Noisify
from src.data_containers.model_hydrogen import HydrogenAnsatz, hydrogen_observable_measurement
from src.processors.processor_quantum import QPU
from src.processors.processor_classic import CPU
from src.error_mitigation import SingleQubitPauliChannel, TwoQubitPauliChannel
from src.main import get_log_experiment


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
    objective = openfermioncirq.HamiltonianObjective(qubit_operator)
    H_observable = objective._hamiltonian_linear_op  # Observable

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


def hamiltonian_density_vs_measure():
    """Plots the H2 expectation value at ground state for different noise models"""
    # Hydrogen ansatz
    ansatz = HydrogenAnsatz()
    # Get optimization
    result = CPU.get_optimized_state(w=ansatz, max_iter=1000)
    parameters = ansatz.operator_parameters
    parameters.update(r=result)
    # Get Hamiltonian objective
    qubit_operator = QPU.get_hamiltonian_evaluation_operator(ansatz)
    # Get Measurement function
    observable_process, circuit_cost = hydrogen_observable_measurement(qubit_operator)

    def get_noise_model(_p: float) -> INoiseModel:
        channel_1q = [SingleQubitPauliChannel(p_x=_p, p_y=_p, p_z=6 * _p)]
        channel_2q = [TwoQubitPauliChannel(p_x=_p, p_y=_p, p_z=6 * _p)]
        return INoiseModel(noise_gates_1q=channel_1q, noise_gates_2q=channel_2q, description=f'asymmetric depolarization (p_tot={16 * _p})')

    # p_list = [float(1/shot) for shot in np.logspace(1, 4, 4)]
    p = 1e-4
    meas_reps = [int(shot) for shot in np.logspace(0, 4, 10)]
    count = 30

    fidelity_list = []
    measure_list = []
    for meas in tqdm(meas_reps):
        noise_model = get_noise_model(_p=0)
        noisy_ansatz = INoiseWrapper(ansatz, noise_model)
        circuit = noisy_ansatz.get_noisy_circuit()
        # Density matrix approach
        exp_fidelity = QPU.get_simulated_noisy_expectation_value(w=noisy_ansatz, r_c=circuit, r=parameters.get_resolved())
        fidelity_list.append(exp_fidelity)
        # Measurement approach
        resolved_circuit = QPU.get_resolved_circuit(circuit, parameters)
        exp_measure = np.array([observable_process(resolved_circuit, noise_model.get_operators, ansatz.qubits, meas) for i in range(count)])
        measure_list.append(exp_measure)

    fidelity_n_list = []
    measure_n_list = []
    for meas in tqdm(meas_reps):
        noise_model = get_noise_model(_p=p)
        noisy_ansatz = INoiseWrapper(ansatz, noise_model)
        circuit = noisy_ansatz.get_noisy_circuit()
        # Density matrix approach
        exp_fidelity = QPU.get_simulated_noisy_expectation_value(w=noisy_ansatz, r_c=circuit, r=parameters.get_resolved())
        fidelity_n_list.append(exp_fidelity)
        # Measurement approach
        resolved_circuit = QPU.get_resolved_circuit(circuit, parameters)
        exp_measure = np.array([observable_process(resolved_circuit, noise_model.get_operators, ansatz.qubits, meas) for i in range(count)])
        measure_n_list.append(exp_measure)

    # Values
    x = np.log10(np.array(meas_reps))
    y = [np.mean(measure) for measure in measure_list]
    y_var = [np.std(measure) for measure in measure_list]
    y_n = [np.mean(measure) for measure in measure_n_list]
    y_var_n = [np.std(measure) for measure in measure_n_list]

    min_diff = abs(y[-1] - fidelity_list[-1])
    min_diff_n = abs(y_n[-1] - fidelity_n_list[-1])

    plt.plot(x, fidelity_list, 'o', label='Density matrix approach')
    plt.errorbar(x, y, yerr=y_var, linestyle='-', marker='^', label=f'Measurement approach')  #  e={round(min_diff,6)}
    plt.plot(x, fidelity_n_list, 'o', label='Density matrix approach (noisy)')
    plt.errorbar(x, y_n, yerr=y_var_n, linestyle='-', marker='^', label=f'Measurement approach (noisy)')  #  e={round(min_diff_n,6)}
    plt.title(f'H2 ansatz circuit comparing density matrix vs measurement observable (Asymmetric Pauli noise p={p})')
    plt.xlabel('log10(average measurement repetition)')
    plt.ylabel('Expectation value for H2 ground state')

    plt.legend(frameon=False)


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
    hamiltonian_density_vs_measure()


if __name__ == '__main__':
    print('Hello World')

    testing()
    plt.show()
