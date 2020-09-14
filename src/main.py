from src.data_containers.model_hydrogen import HydrogenAnsatz, NoisyHydrogen
from src.processors.processor_quantum import QPU
from src.processors.processor_classic import CPU
from src.calculate_minimisation import calculate_and_store
from src.plot_minimisation import read_and_plot
from src.circuit_noise_extension import Noisify


if __name__ == '__main__':

    print("H2-Molecule example. Finding expectation value for molecule with optimal atom distance (.7414 angstrom)")
    print("Using a generated operator tree based on the UCCSD theorem")
    uccsd_ansatz = HydrogenAnsatz()
    parameters = uccsd_ansatz.operator_parameters

    # Get resolved circuit
    print('Show circuit with initial state preparation:')
    circuit = QPU.get_initial_state_circuit(uccsd_ansatz)
    circuit.append(uccsd_ansatz.circuit)
    print(circuit)

    # Get variational study
    result = CPU.get_optimized_state(w=uccsd_ansatz, max_iter=1000)
    print(f'Operator expectation value: {result.optimal_value}\nOperator parameters: {result.optimal_parameters}')

    parameters.update(r=result)
    resolved_circuit = QPU.get_resolved_circuit(circuit, parameters)
    print(resolved_circuit)

    # --------------------------

    # # First test in applying noise to circuits or molecule classes directly
    # print("\nTaking the existing circuit and apply a noise filter:")
    # # Noisy circuit
    # clean_hydrogen = HydrogenAnsatz()
    # noise_circuit = QPU.get_initial_state_circuit(clean_hydrogen)  # Append state initialisation
    # noise_circuit.append(clean_hydrogen.circuit)  # Append operations
    # noise_circuit = Noisify.introduce_noise(noise_circuit)  # 'Noisify'
    # print(noise_circuit)
    #
    # print("\nIntroduce noise directly into the initial state and operator preparations of the Hydrogen model:")
    # # Noisy initial state
    # noisy_hydrogen = NoisyHydrogen()
    # noise_circuit = cirq.Circuit(noisy_hydrogen.initial_state(noisy_hydrogen.qubits))
    # noise_circuit.append(noisy_hydrogen.operations(noisy_hydrogen.qubits))
    # print(noise_circuit)

    # --------------------------

    filename = 'H2_semi_minimisation'
    # Run 'optimization and store data in a json format
    wave_function_class = HydrogenAnsatz()
    calculate_and_store(wave_class=wave_function_class, filename=filename)
    # Read stored json format and express as a good old matlab plot
    plt_obj = read_and_plot(filename=filename)
    plt_obj.show()

    # --------------------------

