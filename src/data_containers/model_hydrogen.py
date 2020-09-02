from typing import Sequence
import cirq
import numpy as np
from openfermion import MolecularData

from src.data_containers.helper_interfaces.i_wave_function import IGeneralizedUCCSD
from src.data_containers.helper_interfaces.i_parameter import IParameter


class HydrogenAnsatz(IGeneralizedUCCSD):

    def __init__(self):
        molecule = IParameter({'r0': 1.5})  # .7414, 'r1': 1.
        super().__init__(molecule)

    # IWaveFunction
    def _generate_molecule(self, p: IParameter) -> MolecularData:
        """Produce molecule that can be used by the hamiltonian.
        Using a singlet state with S = 0 to specify we are looking for the lowest singlet energy state.
        multiplicity = 2S + 1
        """
        geometry = [('H', (0., 0., 0.)), ('H', (0., 0., p['r0']))]  # , ('H', (0., p['r1'], 0.))
        molecule = MolecularData(geometry=geometry, basis='sto-3g', multiplicity=1, charge=0, description=format(p['r0']))
        molecule.load()
        return molecule

    # IWaveFunction
    def initial_state(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        """
        Initial state representation of the Hartree Fock ansatz.
        |Phi> = |0011> + |1100> + |1001> + |0110> (disregarding normalization)
        Start with |0000>. Example,
        Apply two Hadamard (on q1 and q2) to get a 4 state superposition.
        Apply two X (on q0 and q3).
        Apply two CNOT (q1 to q3 and q2 to q0).
        :param qubits: Circuit qubits, (0, 1, 2, 3)
        :return:  X_0 H_1 H_2 X_3 CNOT_13 CNOT_20
        """

        yield [cirq.rx(np.pi / 2).on(qubits[0]),
               cirq.H.on(qubits[1]),
               cirq.H.on(qubits[2]),
               cirq.rx(np.pi / 2).on(qubits[3])]
        yield [cirq.CNOT(qubits[1], qubits[3]),
               cirq.CNOT(qubits[2], qubits[0])]
