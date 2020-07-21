from typing import Sequence
import cirq
import numpy as np
from openfermion import MolecularData

from src.data_containers.helper_interfaces.i_wave_function import IWaveFunction
from src.data_containers.helper_interfaces.i_parameter import IParameter


class HydrogenAnsatz(IWaveFunction):

    # VariationalAnsatz
    def _generate_qubits(self) -> Sequence[cirq.Qid]:
        """Produce qubits that can be used by the ansatz circuit"""
        return [cirq.GridQubit(i, j) for i in range(2) for j in range(2)]

    # IWaveFunction
    def _generate_parameters(self) -> IParameter:
        """Produce parameters used to define the operations in the ansatz circuit."""
        return IParameter({'alpha': 1})

    # IWaveFunction
    def _generate_molecule(self) -> MolecularData:
        """Produce molecule that can be used by the hamiltonian."""
        r = .7414
        geometry = [('H', (0., 0., 0.)), ('H', (0., 0., r))]
        return MolecularData(geometry=geometry, basis='sto-3g', multiplicity=1, description=format(r))

    # VariationalAnsatz
    def operations(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        """Produce the operations of the ansatz circuit.
        The operations should use Symbols produced by the `params` method
        of the ansatz.
        """
        _parameters = list(self.params())
        yield [cirq.ry(np.pi).on(qubits[0]),
               cirq.ry(np.pi).on(qubits[1])]
        yield [cirq.rx(np.pi / 2).on(qubits[0]),
               cirq.ry(np.pi / 2).on(qubits[1]),
               cirq.ry(np.pi / 2).on(qubits[2]),
               cirq.ry(np.pi / 2).on(qubits[3])]
        yield [cirq.CNOT(qubits[0], qubits[1]),
               cirq.CNOT(qubits[1], qubits[2]),
               cirq.CNOT(qubits[2], qubits[3])]
        yield cirq.rz(_parameters[0]).on(qubits[3])
        yield [cirq.CNOT(qubits[2], qubits[3]),
               cirq.CNOT(qubits[1], qubits[2]),
               cirq.CNOT(qubits[0], qubits[1])]
        yield [cirq.rx(-np.pi / 2).on(qubits[0]),
               cirq.ry(-np.pi / 2).on(qubits[1]),
               cirq.ry(-np.pi / 2).on(qubits[2]),
               cirq.ry(-np.pi / 2).on(qubits[3])]
