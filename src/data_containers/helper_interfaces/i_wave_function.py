from abc import abstractmethod
from typing import Iterable, Sequence
import cirq
import sympy
from openfermioncirq import VariationalAnsatz
from openfermion import MolecularData
from openfermionpsi4 import run_psi4

from src.data_containers.helper_interfaces.i_parameter import IParameter


class IMolecule:

    def get_molecule_params(self) -> IParameter:
        """Produce parameters used to define the molecule structure."""
        return self._params_molecule

    molecule_parameters = property(get_molecule_params)

    def __init__(self, molecule_params: IParameter):
        self._params_molecule = molecule_params
        self.molecule = self.generate_molecule(p=molecule_params)

    @abstractmethod
    def generate_molecule(self, p: IParameter) -> MolecularData:
        """Produce molecule that can be used by the hamiltonian."""
        raise NotImplemented

    def update_molecule(self, p: IParameter) -> MolecularData:
        """Updates molecule parameters and overrides the local data"""
        self._params_molecule = p
        self.molecule = self.generate_molecule(p=p)
        if os.path.exists(self.molecule.filename + '.hdf5'):
            self.molecule.load()
        else:
            self.molecule = run_psi4(self.molecule, run_scf=1, run_mp2=0, run_cisd=0, run_ccsd=0, run_fci=0, verbose=1, tolerate_error=1)
        self.molecule.save()
        return self.molecule


class IWaveFunction(VariationalAnsatz, IMolecule):

    def get_operator_params(self) -> IParameter:
        """Produce parameters used to define the operations in the ansatz circuit."""
        return self._params_operator

    operator_parameters = property(get_operator_params)

    # VariationalAnsatz
    def params(self) -> Iterable[sympy.Symbol]:
        return [sympy.Symbol(key) for key in self._params_operator]

    def __init__(self, operator_params: IParameter, molecule_params: IParameter):
        self._params_operator = operator_params
        IMolecule.__init__(self, molecule_params)
        VariationalAnsatz.__init__(self)

    # VariationalAnsatz
    @abstractmethod
    def _generate_qubits(self) -> Sequence[cirq.Qid]:
        """Produce qubits that can be used by the ansatz circuit."""
        raise NotImplemented

    # VariationalAnsatz
    @abstractmethod
    def operations(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        """Produce the operations of the ansatz circuit.
        The operations should use Symbols produced by the `params` method
        of the ansatz.
        """
        raise NotImplemented

    @abstractmethod
    def initial_state(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        """Produce the initial state of the ansatz circuit before operation.
        The operations should use Symbols produced by the `params` method
        of the ansatz.
        """
        yield []
