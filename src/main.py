import cirq

from src.data_containers.model_hydrogen import HydrogenAnsatz, NoisyHydrogen
from src.processors.processor_quantum import QPU
from src.processors.processor_classic import CPU
from src.circuit_noise_extension import Noisify


if __name__ == '__main__':

    print("H2-Molecule example. Finding expectation value for molecule with optimal atom distance (.7414 angstrom)")
    print("Using a generated operator tree based on the UCCSD theorem")
    uccsd_ansatz = NoisyHydrogen()
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

    print("\nTaking the existing circuit and apply a noise filter:")
    # Noisy circuit
    clean_hydrogen = HydrogenAnsatz()
    noise_circuit = QPU.get_initial_state_circuit(clean_hydrogen)  # Append state initialisation
    noise_circuit.append(clean_hydrogen.circuit)  # Append operations
    noise_circuit = Noisify.introduce_noise(noise_circuit)  # 'Noisify'
    print(noise_circuit)

    print("\nIntroduce noise directly into the initial state and operator preparations of the Hydrogen model:")
    # Noisy initial state
    noisy_hydrogen = NoisyHydrogen()
    noise_circuit = cirq.Circuit(noisy_hydrogen.initial_state(noisy_hydrogen.qubits))
    noise_circuit.append(noisy_hydrogen.operations(noisy_hydrogen.qubits))
    print(noise_circuit)

    # --------------------------

    # CPU.get_optimized_ground_state(w=uccsd_ansatz, qpu_iter=10, cpu_iter=10)

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
