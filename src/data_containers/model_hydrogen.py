from typing import Sequence
import cirq
import numpy as np
from openfermion import MolecularData

from src.data_containers.helper_interfaces.i_wave_function import IGeneralizedUCCSD
from src.data_containers.helper_interfaces.i_parameter import IParameter
from src.circuit_noise_extension import Noisify


class HydrogenAnsatz(IGeneralizedUCCSD):

    def __init__(self):
        molecule = IParameter({'r0': .7414})  # .7414, 'r1': 1.
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
        print(f'Reference (Full Configuration Energy: {molecule.fci_energy})')
        return molecule

    # IWaveFunction
    def initial_state(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        """
        Using Hartree Fock state:
        |1100>
        instead of the most general state:
        |0011> + |1100> + |1001> + |0110> (disregarding normalization)

        :param qubits: Circuit qubits, (0, 1, 2, 3)
        :return:  X_0 X_1
        """
        yield [cirq.rx(np.pi).on(qubits[0]),
               cirq.rx(np.pi).on(qubits[1])]


class NoisyHydrogen(HydrogenAnsatz):

    def operations(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        yield Noisify.introduce_noise_tree(IGeneralizedUCCSD.operations(self, qubits=qubits))

    def initial_state(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        yield Noisify.introduce_noise_tree(HydrogenAnsatz.initial_state(self, qubits=qubits))
