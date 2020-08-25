import cirq
# General concept:
# Qubits (cirq.gridQubit) are 2-level quantum systems and are represented by a Qid (cirq.ops.raw_types) object.
#   They seem to be statically assigned (not dependent on circuit class).
# Gates (cirq.ops.common_gates) are operator functions that work on qubits (cirq.gridQubit).
# Operations (cirq.GateOperation) are gate-qubit pairs.
# Circuits (cirq.circuits.circuit) are an ordered set of Moment (cirq.ops.moment) objects.
#   They are not collections of qubits and operations.
# Difference between run (mimics actual hardware) and simulation (allows for amplitude access etc.).


def define_circuit() -> cirq.circuits.circuit:
    """Returns a basic cirq defined circuit"""
    return cirq.Circuit()


def gate_on_qubits(gate: cirq.ops.common_gates, qubits: [cirq.GridQubit]) -> [cirq.GateOperation]:
    """Performs single gate on an array of grid qubits"""
    for qubit in qubits:
        yield gate(qubit)


def gates_on_qubits(gates: [cirq.ops.common_gates], qubits: [cirq.GridQubit]) -> [cirq.GateOperation]:
    """Performs an array of gates on an array of grid qubits"""
    for gate in gates:
        yield gate_on_qubits(gate, qubits)


def sub_circuit_column(operations: [cirq.GateOperation]) -> cirq.ops.moment:
    """Groups an array of operations (gate on qubit) in a single circuit moment (column)"""
    yield cirq.Moment(operations)


def measure_column(qubits: [cirq.GridQubit], basis: str) -> cirq.ops.moment:
    """
    Performs a measurement on grid qubits in specific basis
    :param qubits: Grid qubits to be measured
    :param basis: (Pauli) basis (x, y, z)
    :return: Operation targeting qubit measurement
    """
    yield cirq.measure(*qubits, key=basis)
