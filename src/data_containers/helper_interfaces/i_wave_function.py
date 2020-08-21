from abc import abstractmethod
from typing import Iterable, Sequence
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

    def _get_hamiltonian(self) -> of.InteractionOperator:
        """Returns second-quantization Hamiltonian"""
        return self.molecule.get_molecular_hamiltonian()

    # hamiltonian = property(_get_hamiltonian)

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
        # UCC-Single amplitudes
        _single_amp = self.molecule.ccsd_single_amps
        # UCC-Double amplitudes
        _double_amp = self.molecule.ccsd_double_amps
        # Creation/Annihilation operators
        _ucc_operator = of.normal_ordered(of.uccsd_generator(_single_amp, _double_amp))
        # for key in _ucc_operator.terms.keys():
        #     _ucc_operator.terms[key] *= _symbols[0]  # sympy.Expr(_parameters[0])
        # Pauli string operators
        _qubit_operator = of.transforms.jordan_wigner(_ucc_operator)
        _gate_operators = list()
        # for key in _qubit_operator.terms.keys():
        #     # Tuple of (qubit ID, rotation ID)
        #     for i, (q_ID, r_ID) in enumerate(key):
        #         if r_ID == 'X':
        #             _gate_operators.append(cirq.rx(_symbols[0]).on(qubits[q_ID]))  # * _qubit_operator.terms[key]
        #         if r_ID == 'Y':
        #             _gate_operators.append(cirq.ry(_symbols[0]).on(qubits[q_ID]))
        #         if r_ID == 'Z':
        #             _gate_operators.append(cirq.rz(_symbols[0]).on(qubits[q_ID]))
        IGeneralizedUCCSD.qubit_to_gate_operators(qubit_operators=[_qubit_operator], qubits=qubits)
        # yield [cirq.rx(np.pi / 2).on(qubits[0]),
        #        cirq.ry(np.pi / 2).on(qubits[1]),
        #        cirq.ry(np.pi / 2).on(qubits[2]),
        #        cirq.ry(np.pi / 2).on(qubits[3])]
        # yield [cirq.CNOT(qubits[0], qubits[1]),
        #        cirq.CNOT(qubits[1], qubits[2]),
        #        cirq.CNOT(qubits[2], qubits[3])]
        yield _gate_operators
        # yield [cirq.CNOT(qubits[2], qubits[3]),
        #        cirq.CNOT(qubits[1], qubits[2]),
        #        cirq.CNOT(qubits[0], qubits[1])]
        # yield [cirq.rx(-np.pi / 2).on(qubits[0]),
        #        cirq.ry(-np.pi / 2).on(qubits[1]),
        #        cirq.ry(-np.pi / 2).on(qubits[2]),
        #        cirq.ry(-np.pi / 2).on(qubits[3])]

    @staticmethod
    def qubit_to_gate_operators(qubit_operators: [of.QubitOperator], qubits: Sequence[cirq.Qid]) -> [cirq.GateOperation]:
        # dictionary for converting x and y rotations to z
        rot_dic = {'X': lambda q, inv: cirq.H.on(q),
                   'Y': lambda q, inv: cirq.rx(-inv * np.pi / 2).on(q),
                   'Z': lambda q, inv: cirq.I.on(q)}

        # list of parameters
        parameters = [sympy.Symbol('alpha%d' % i) for i in range(int(.5 * len(qubit_operators)))]

        for q_op in qubit_operators:
            for key, value in q_op.terms:
                # Tuple of (qubit ID, rotation ID)
                # pauli_strings = key
                # Get sign of Pauli string
                sign = np.sign(value)
                for qbt, pau in key:
                    yield rot_dic[pau](qubit_list[qbt], 1)
        pass

    def initial_state(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        """Returns initial state representation of the Hartree Fock ansatz"""
        yield [cirq.rz(np.pi).on(qubits[i]) for i in range(len(qubits))]

    # def FOtoCIRQ(fo_list, qubit_list):
    #     '''
    #     Function to convert a list of FermionicOperators to a cirq.Circuit()
    #
    #     Input:
    #     fo_list [list or FermionicOperator] : FermionicOperator or list of FermionicOperators
    #     qubit_list [list]                   : list of cirq.LineQubits
    #
    #     '''
    #
    #     # check that fo_list is either a FermionOperator or a list of FermionOperators
    #     if not isinstance(fo_list, (list, openfermion.FermionOperator)):
    #         raise TypeError('Input must be a list or QubitOperator')
    #     if isinstance(fo_list, openfermion.FermionOperator):
    #         fo_list = list(fo_list)
    #
    #     # dictionary for converting x and y rotations to z
    #     rot_dic = {'X': lambda q, inv: cirq.H.on(q),
    #                'Y': lambda q, inv: cirq.rx(-inv * numpy.pi / 2).on(q),
    #                'Z': lambda q, inv: cirq.I.on(q)}
    #
    #     # list of parameters
    #     parameters = [Symbol('theta%d' % i) for i in range(int(.5 * len(fo_list)))]
    #
    #     for i, fo in enumerate(fo_list[::2]):
    #         # Jordan-Wigner transform each fermionic operator + its Hermitian conjugate
    #         qo_list = list(openfermion.jordan_wigner(fo + openfermion.hermitian_conjugated(fo)))
    #
    #         for qo in qo_list:
    #             # get tuple of tuples containing which Pauli on which qubit appears in the Pauli string
    #             # ex. [X0 Y1] -> ((0, X), (1, Y))
    #             paulis = list(qo.terms.keys())[0]
    #
    #             # get sign of Pauli string
    #             sign = numpy.sign(list(qo.terms.values())[0])
    #
    #             # convert Pauli string into rz and CNOT gates
    #             # see Whitfield paper for details
    #             for qbt, pau in paulis:
    #                 yield rot_dic[pau](qubit_list[qbt], 1)
    #
    #             if len(paulis) > 1:
    #                 for j in range(len(paulis) - 1):
    #                     yield cirq.CNOT(qubit_list[paulis[j][0]],
    #                                     qubit_list[paulis[j + 1][0]])
    #
    #             # the factor of 2 is mostly for convention, since we're optimizing the parameters
    #             # see Whitfield paper for details
    #             yield cirq.rz(2 * sign * parameters[i]).on(qubit_list[paulis[-1][0]])
    #
    #             if len(paulis) > 1:
    #                 for j in range(len(paulis) - 1, 0, -1):
    #                     yield cirq.CNOT(qubit_list[paulis[j - 1][0]],
    #                                     qubit_list[paulis[j][0]])
    #
    #             for qbt, pau in paulis:
    #                 yield rot_dic[pau](qubit_list[qbt], -1)

