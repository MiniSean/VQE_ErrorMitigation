import numpy as np
import cirq
import openfermioncirq
from openfermioncirq.optimization import OptimizationTrialResult

from src.processors.processor_quantum import QPU
from src.data_containers.helper_interfaces.i_hamiltonian import IHamiltonian
from src.data_containers.helper_interfaces.i_wave_function import IWaveFunction


class CPU:

    @staticmethod
    def get_optimized_ground_state():
        pass

    @staticmethod
    def get_optimized_state(w: IWaveFunction, max_iter: int) -> OptimizationTrialResult:
        initial_state_circuit = QPU.get_initial_state_circuit(w=w)
        study = IHamiltonian.get_variational_study(w=w, p_c=initial_state_circuit, name='HydrogenStudy')
        return QPU.get_optimized_state(s=study, max_iter=max_iter)
