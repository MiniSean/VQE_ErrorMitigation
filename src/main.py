import cirq
from src.data_containers.model_hydrogen import HydrogenAnsatz
from src.processors.processor_quantum import QPU
from src.processors.processor_classic import CPU
from src.data_containers.helper_interfaces.i_hamiltonian import IHamiltonian


if __name__ == '__main__':

    ansatz = HydrogenAnsatz()
    parameters = ansatz.parameters

    # Get resolved circuit
    circuit = ansatz.circuit
    circuit.append(cirq.measure(*ansatz.qubits, key='x'))  # Temporary forced measurement on x axis
    resolved_circuit = QPU.get_resolved_circuit(circuit, parameters)
    print(circuit)

    # Get variational study
    study = IHamiltonian.get_variational_study(w=ansatz, name='HydrogenStudy')
    result = QPU.get_optimization_result(s=study, max_iter=10)
    print(f'Operator expectation value: {result.optimal_value}, with parameters: {result.optimal_parameters}')
