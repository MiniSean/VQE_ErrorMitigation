from typing import Sequence
import cirq
import numpy as np
import os
from openfermion import MolecularData
from openfermionpsi4 import run_psi4

from src.data_containers.helper_interfaces.i_wave_function import IGeneralizedUCCSD
from src.data_containers.helper_interfaces.i_parameter import IParameter
from src.circuit_noise_extension import Noisify


class HydrogenAnsatz(IGeneralizedUCCSD):

    def __init__(self):
        molecule = IParameter({'r0': .7414})  # .7414, 'r1': 1.
        super().__init__(molecule)

    def _generate_molecule(self, p: IParameter) -> MolecularData:
        """Produce molecule that can be used by the hamiltonian.
        Using a singlet state with S = 0 to specify we are looking for the lowest singlet energy state.
        multiplicity = 2S + 1
        """
        geometry = [('H', (0., 0., 0.)), ('H', (0., 0., p['r0']))]  # , ('H', (0., p['r1'], 0.))
        basis = 'sto-3g'
        multiplicity = 1
        charge = 0
        description = str(p['r0'])

        # Change to whatever directory you want
        cwd = os.getcwd()
        data_directory = cwd+'/mol_data'

        if not os.path.exists(data_directory):
            os.mkdir(data_directory)

        filename = data_directory+'/H2_'+description

        run_scf = 1
        run_mp2 = 1
        run_cisd = 1
        run_ccsd = 1
        run_fci = 1
        delete_input = False
        delete_output = False
        verbose = False

        molecule = MolecularData(
            geometry,
            basis,
            multiplicity,
            description=description,
            filename=filename)

        if os.path.exists('{}.hdf5'.format(filename)):
            molecule.load()

        else:
            molecule = run_psi4(molecule,
                                verbose=verbose,
                                run_scf=run_scf,
                                run_mp2=run_mp2,
                                run_cisd=run_cisd,
                                run_ccsd=run_ccsd,
                                run_fci=run_fci)

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
