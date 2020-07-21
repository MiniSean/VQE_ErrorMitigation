from abc import abstractmethod
from typing import Iterable, Sequence
import cirq
import sympy
from openfermioncirq import VariationalAnsatz
from openfermion import MolecularData

from src.data_containers.helper_interfaces.i_parameter import IParameter


class IWaveFunction(VariationalAnsatz):

    @abstractmethod
    def _generate_parameters(self) -> IParameter:
        """Produce parameters used to define the operations in the ansatz circuit."""
        raise NotImplemented

    # VariationalAnsatz
    def params(self) -> Iterable[sympy.Symbol]:
        return [sympy.Symbol(key) for key in self.parameters]

    def __init__(self):
        self.parameters = self._generate_parameters()
        self.molecule = self._generate_molecule()
        super().__init__()

    # VariationalAnsatz
    @abstractmethod
    def _generate_qubits(self) -> Sequence[cirq.Qid]:
        """Produce qubits that can be used by the ansatz circuit."""
        raise NotImplemented

    @abstractmethod
    def _generate_molecule(self) -> MolecularData:
        """Produce molecule that can be used by the hamiltonian."""
        raise NotImplemented

    # VariationalAnsatz
    @abstractmethod
    def operations(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        """Produce the operations of the ansatz circuit.
        The operations should use Symbols produced by the `params` method
        of the ansatz.
        """
        raise NotImplemented




