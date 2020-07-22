import cirq
from src.data_containers.model_hydrogen import HydrogenAnsatz
from src.processors.processor_quantum import QPU
from src.processors.processor_classic import CPU


if __name__ == '__main__':

    ansatz = HydrogenAnsatz()
    parameters = ansatz.parameters

    # Get resolved circuit
    circuit = ansatz.circuit
    circuit.append(cirq.measure(*ansatz.qubits, key='x'))  # Temporary forced measurement on x axis
    resolved_circuit = QPU.get_resolved_circuit(circuit, parameters)
    print(resolved_circuit)

    # Get variational study
    result = CPU.get_optimized_state(w=ansatz, max_iter=10)
    print(f'Operator expectation value: {result.optimal_value}\nOperator parameters: {result.optimal_parameters}')

    parameters.update(r=result)
    resolved_circuit = QPU.get_resolved_circuit(circuit, parameters)
    print(resolved_circuit)
