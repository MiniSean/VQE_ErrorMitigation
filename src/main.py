import cirq

from src.data_containers.model_hydrogen import HydrogenAnsatz
from src.data_containers.helper_interfaces.i_noise_wrapper import INoiseWrapper, INoiseModel
from src.error_mitigation import SingleQubitPauliChannel, TwoQubitPauliChannel, simulate_error_mitigation
from src.processors.processor_quantum import QPU
from src.processors.processor_classic import CPU


def basic_ansatz():
    print("H2-Molecule example. Finding expectation value for molecule with optimal atom distance (.7414 angstrom)")
    print("Using a generated operator tree based on the UCCSD theorem")
    uccsd_ansatz = HydrogenAnsatz()
    parameters = uccsd_ansatz.operator_parameters

    # Get resolved circuit
    print('Show circuit with initial state preparation:')
    circuit = QPU.get_initial_state_circuit(uccsd_ansatz)
    circuit.append(uccsd_ansatz.circuit)
    # noise_channel = INoiseModel(noise_gates=[cirq.depolarize(p=.5)], description=f'Depolarize (p={0.5})')
    # circuit = Noisify.introduce_noise(circuit, noise_channel.get_callable())
    print(circuit)

    # Get variational study
    result = CPU.get_optimized_state(w=uccsd_ansatz, max_iter=1000)
    print(f'Operator expectation value: {result.optimal_value}\nOperator parameters: {result.optimal_parameters}')

    parameters.update(r=result)
    resolved_circuit = QPU.get_resolved_circuit(circuit, parameters)
    print(resolved_circuit)


def custom_ansatz():
    print("H2-Molecule example. Finding expectation value for molecule with optimal atom distance (.7414 angstrom)")
    print("Using a generated operator tree based on the UCCSD theorem")
    uccsd_ansatz = HydrogenAnsatz()
    parameters = uccsd_ansatz.operator_parameters

    # Get custom optimization
    noise_channel = INoiseModel(noise_gates_1q=[cirq.bit_flip(p=.5), cirq.phase_flip(p=.5)], noise_gates_2q=[], description=f'Bit and Phase flip (p={0.5})')  # [cirq.AmplitudeDampingChannel(gamma=.1)]
    noisy_ansatz = INoiseWrapper(uccsd_ansatz, noise_channel)
    values, params = CPU.get_custom_optimized_state(n_w=noisy_ansatz, max_iter=10)
    print(f'Operator expectation value: {values}\nOperator parameters: {params}')

    for i, key in enumerate(parameters.dict.keys()):  # Dirty set parameter values
        parameters[key] = params[i]
    resolved_circuit = QPU.get_resolved_circuit(circuit, parameters)
    print(resolved_circuit)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import openfermioncirq
    uccsd_ansatz = HydrogenAnsatz()
    parameters = uccsd_ansatz.operator_parameters

    # --------------------------

    print(f'Circuit optimized without noise. Evaluated with noise.')
    # Perform error mitigation on Hydrogen ansatz
    p = 1e-4

    # Construct noise model
    channel_1q = [SingleQubitPauliChannel(p_x=p, p_y=p, p_z=6 * p)]
    channel_2q = [TwoQubitPauliChannel(p_x=p, p_y=p, p_z=6 * p)]
    noise_model = INoiseModel(noise_gates_1q=channel_1q, noise_gates_2q=channel_2q, description=f'asymmetric depolarization (p_tot={16 * p})')
    noisy_ansatz = INoiseWrapper(uccsd_ansatz, noise_model)

    # Construct circuit
    circuit = noisy_ansatz.get_clean_circuit()
    print(circuit)
    # Get variational study
    result = CPU.get_optimized_state(w=uccsd_ansatz, max_iter=1000)
    print(f'Operator expectation value: {result.optimal_value}\nOperator parameters: {result.optimal_parameters}')
    parameters.update(r=result)
    resolved_circuit = QPU.get_resolved_circuit(circuit, parameters)
    print(resolved_circuit)

    # Get Hamiltonian objective
    qubit_operator = QPU.get_hamiltonian_evaluation_operator(uccsd_ansatz)
    objective = openfermioncirq.HamiltonianObjective(qubit_operator)
    H_operator = objective._hamiltonian_linear_op  # Observable

    # Plot error mitigation
    simulate_error_mitigation(clean_circuit=resolved_circuit, noise_model=noise_model, process_circuit_count=10000, density_representation=True, hamiltonian_objective=H_operator)
    plt.show()
