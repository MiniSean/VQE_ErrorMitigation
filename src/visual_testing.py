# https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec
import numpy as np
import pylatexenc
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix
from qiskit.visualization import plot_state_city, circuit_drawer, plot_state_hinton
import cirq
import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure, Subplot
from typing import List, Tuple, Union
from src.error_mitigation import BasisUnitarySet, IBasisGateSingle, reconstruct_from_basis
from src.data_containers.helper_interfaces.i_noise_wrapper import INoiseModel, INoiseWrapper
from src.circuit_noise_extension import Noisify
from src.data_containers.model_hydrogen import HydrogenAnsatz
from src.processors.processor_quantum import QPU
from src.processors.processor_classic import CPU


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

    matrix = BasisUnitarySet.gate_set_tomography(gate=effective_unitary)[1]
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

    matrix = BasisUnitarySet.gate_set_tomography(gate=effective_unitary)[1]
    # Plot
    setup_before_after_2D_3D_plot(density_matrix=effective_unitary, noise_matrix=matrix, circuit=None, title=r'Gate set tomography of Density Matrix')

    print(effective_unitary)
    print(matrix)

    qp, basis = BasisUnitarySet.get_decomposition(gate=effective_unitary)
    goal = reconstruct_from_basis(qp, basis)

    print(qp)


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
    gst_identity_ideal()


if __name__ == '__main__':
    print('Hello World')

    testing()
    plt.show()
