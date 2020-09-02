import cirq

from src.data_containers.model_hydrogen import HydrogenAnsatz
from src.processors.processor_quantum import QPU
from src.processors.processor_classic import CPU
from src.circuit_noise_extension import Noisify


if __name__ == '__main__':

    print("H2-Molecule example. Finding expectation value for molecule with optimal atom distance (.7414 angstrom)")
    print("Using a generated operator tree based on the UCCSD theorem")
    uccsd_ansatz = HydrogenAnsatz()
    parameters = uccsd_ansatz.operator_parameters

    # Get resolved circuit
    print('Show circuit with initial state preparation')
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

    # Noisy circuit
    noise_circuit = Noisify.introduce_noise(circuit)
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
