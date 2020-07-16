import cirq
from src.data_containers.qubit_grid import define_square_grid
from src.data_containers.qubit_circuit import define_circuit, gate_on_qubits, gates_on_qubits, sub_circuit_column, measure_column


if __name__ == '__main__':
    # Define qubits and circuit
    circuit = define_circuit()
    qubits = define_square_grid(2)
    # Define gates
    gate_x = cirq.XPowGate(exponent=.5)
    gate_y = cirq.YPowGate(exponent=.5)
    # Create operation layers
    operations_x = gate_on_qubits(gate_x, qubits[0:2])
    operations_y = gate_on_qubits(gate_y, qubits[2:4])
    moment_identity = sub_circuit_column([operations_x, operations_y])
    circuit.append(moment_identity)
    circuit.append(measure_column(qubits, basis='x'))

    print(circuit)

    results = cirq.Simulator().run(circuit, repetitions=10)
    print(results.histogram(key='x'))
