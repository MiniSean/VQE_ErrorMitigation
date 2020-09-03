from abc import abstractmethod
from typing import Iterable, Sequence, Dict, Callable
import os
import cirq
import sympy
import numpy as np
from openfermioncirq import VariationalAnsatz
import openfermion as of
from openfermionpsi4 import run_psi4
# from openfermionpyscf import run_pyscf

from src.data_containers.helper_interfaces.i_parameter import IParameter


class IMolecule:

    def get_molecule_params(self) -> IParameter:
        """Produce parameters used to define the molecule structure."""
        return self._params_molecule

    molecule_parameters = property(get_molecule_params)

    def __init__(self, molecule_params: IParameter):
        self._params_molecule = molecule_params
        self.molecule = self._generate_molecule(p=molecule_params)
        # self.molecule = run_pyscf(self.molecule, run_mp2=True, run_cisd=True, run_ccsd=True, run_fci=True)
        # two_electron_integral = self.molecule.two_body_integrals

    @abstractmethod
    def _generate_molecule(self, p: IParameter) -> of.MolecularData:
        """Produce molecule that can be translated to a hamiltonian."""
        raise NotImplemented

    def update_molecule(self, p: IParameter) -> of.MolecularData:
        """Updates molecule parameters and overrides the local data"""
        self._params_molecule = p
        self.molecule = self._generate_molecule(p=p)
        self.molecule.save()
        # if os.path.exists(self.molecule.filename + '.hdf5'):
        #     self.molecule.load()
        # else:
        # self.molecule = run_psi4(self.molecule, run_mp2=True, run_cisd=True, run_ccsd=True, run_fci=True)
        # two_electron_integral = self.molecule.two_body_integrals
        # self.molecule.save()
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


class IGeneralizedUCCSD(IWaveFunction):
    """
    "For experimental addressability, the qubits must, in general, be distinguishable. However, the electrons of the molecular system are indistinguishable.
    The Jordan-Wigner transform is used to circumvent this issue" - Whitfield paper
    """

    def _get_hamiltonian(self) -> of.InteractionOperator:
        """Returns second-quantization Hamiltonian"""
        return self.molecule.get_molecular_hamiltonian()

    def _get_fermionic_operators(self) -> [of.FermionOperator]:
        """Returns second-quantization Hamiltonian in fermionic operator form"""
        # UCC-Single amplitudes
        _single_amp = self.molecule.ccsd_single_amps
        # UCC-Double amplitudes
        _double_amp = self.molecule.ccsd_double_amps
        # Creation/Annihilation operators
        _ucc_operator = of.normal_ordered(of.uccsd_generator(_single_amp, _double_amp))
        # check that _ucc_operator is either a FermionOperator or a list of FermionOperators
        if isinstance(_ucc_operator, of.FermionOperator):
            _ucc_operator = list(_ucc_operator)
        return _ucc_operator

    # hamiltonian = property(_get_hamiltonian)

    def __init__(self, molecule_params: IParameter):
        IMolecule.__init__(self, molecule_params)
        parameter_count = int(.5 * len(self._get_fermionic_operators()))
        self._params_operator = IParameter({f'alpha{i}': 0.0 for i in range(parameter_count)})
        VariationalAnsatz.__init__(self)

    @abstractmethod
    def _generate_molecule(self, p: IParameter) -> of.MolecularData:
        """Produce molecule that can be translated to a hamiltonian"""
        raise NotImplemented

    def _generate_qubits(self) -> Sequence[cirq.Qid]:
        """Specifies the number of required qubits based on the molecule hamiltonian"""
        qubit_count = of.count_qubits(self._get_hamiltonian())
        row = int(np.ceil(np.sqrt(qubit_count)))
        return [cirq.GridQubit(i, j) for i in range(row) for j in range(row)]

    def operations(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        """Returns qubit operators based on STO-3G and UCCSD ansatz"""
        _symbols = list(self.params())
        _ucc_operator = self._get_fermionic_operators()

        for i, fo in enumerate(_ucc_operator[::2]):
            # Jordan-Wigner transform each fermionic operator + its Hermitian conjugate
            qo_list = list(of.jordan_wigner(fo - of.hermitian_conjugated(fo)))
            yield IGeneralizedUCCSD.qubit_to_gate_operators(qubit_operators=qo_list, qubits=qubits, par=_symbols[i])

    @staticmethod
    def pauli_basis_map() -> Dict[str, Callable[[int, int], cirq.GateOperation]]:
        """
        Transformation basis to move from exponential Pauli matrix tensor to the appropriate unitary basis transformation.
        Recap (Pauli matrices basis/eigenvector):
        Z-basis: |0>, |1>, X-basis: ~(|0> + |1>), ~(|0> - |1>), Y-basis: ~(|0> + i|1>), ~(|0> - i|1>)
        (Where ~ represents the normalization factor). Mapping information:
        Hadamard maps between basis X and Z.
        Y ( R_x(-pi/2) ) = exp(i X pi/4) ) maps between basis Y and X. (With potential global phase)
        (Where Z is taken as conventional basis, therefor I maps arbitrarily between basis Z and Z.)
        :return:
        """
        return {'X': lambda q, inv: cirq.H.on(q),
                'Y': lambda q, inv: cirq.rx(-inv * np.pi / 2).on(q),
                'Z': lambda q, inv: cirq.I.on(q)}

    @staticmethod
    def qubit_to_gate_operators(qubit_operators: [of.QubitOperator], qubits: Sequence[cirq.Qid], par: sympy.Symbol) -> [cirq.GateOperation]:
        """
        Function to convert a list of QubitOperators to a cirq.Circuit().
        :param qubit_operators: list of FermionicOperators
        :param qubits: list of cirq.GridQubits
        :param par: parameter used per QubitOpterator string
        :return: list of GateOperations
        """
        # dictionary for converting x and y rotations to z
        map = IGeneralizedUCCSD.pauli_basis_map()

        for q_op in qubit_operators:
            # Single user pauli string
            # Get tuple of tuples containing which Pauli on which qubit appears in the Pauli string
            pauli_op = list(q_op.terms.keys())[0]
            sign = np.sign(list(q_op.terms.values())[0])

            for qbt, pau in pauli_op:  # Tuple of (qubit ID, rotation ID)
                yield map[pau](qubits[qbt], 1)

            for j in range(len(pauli_op) - 1):
                yield cirq.CNOT(qubits[pauli_op[j][0]], qubits[pauli_op[j + 1][0]])

            # Last operator in Pauli string specifies the R_z qubit focus
            yield cirq.rz(2 * sign * par).on(qubits[pauli_op[-1][0]])

            for j in range(len(pauli_op) - 1, 0, -1):
                yield cirq.CNOT(qubits[pauli_op[j - 1][0]], qubits[pauli_op[j][0]])

            for qbt, pau in pauli_op:
                yield map[pau](qubits[qbt], -1)
            # # Multi user pauli string
            # for pauli_op, coefficient in q_op.terms.items():  # Tuple of (Pauli operator string, exponent coefficient)
            #     # convert Pauli string into rz and CNOT gates
            #     sign = np.sign(coefficient)
            #     for qbt, pau in pauli_op:  # Tuple of (qubit ID, rotation ID)
            #         yield map[pau](qubits[qbt], 1)
            #
            #     for j in range(len(pauli_op) - 1):
            #         yield cirq.CNOT(qubits[pauli_op[j][0]], qubits[pauli_op[j + 1][0]])
            #
            #     # Last operator in Pauli string specifies the R_z qubit focus
            #     yield cirq.rz(2 * sign * par).on(qubits[pauli_op[-1][0]])
            #
            #     for j in range(len(pauli_op) - 1, 0, -1):
            #         yield cirq.CNOT(qubits[pauli_op[j - 1][0]], qubits[pauli_op[j][0]])
            #
            #     for qbt, pau in pauli_op:
            #         yield map[pau](qubits[qbt], -1)

    @abstractmethod
    def initial_state(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        """
        Not Implemented yet
        """
        yield []
