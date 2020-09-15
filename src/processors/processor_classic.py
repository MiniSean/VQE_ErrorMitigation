import numpy as np
from scipy import optimize
from openfermioncirq.optimization import OptimizationTrialResult
from typing import List

from src.processors.processor_quantum import QPU
from src.data_containers.helper_interfaces.i_wave_function import IWaveFunction
from src.data_containers.helper_interfaces.i_noise_wrapper import INoiseWrapper
from src.data_containers.helper_interfaces.i_parameter import IParameter
from src.data_containers.helper_interfaces.i_collection import IContainer


class CPU:

    @staticmethod
    def get_optimized_ground_state(w: IWaveFunction, qpu_iter: int, cpu_iter: int):
        molecule_params = w.molecule_parameters
        optimization_trail = list()

        def optimize_func(x: np.ndarray) -> float:
            for i, key in enumerate(molecule_params):
                molecule_params.dict[key] = x[i]
            w.update_molecule(molecule_params)
            print(molecule_params)

            trial_result = CPU.get_optimized_state(w=w, max_iter=qpu_iter)

            optimization_trail.append(trial_result)
            print(trial_result.optimal_value)
            return trial_result.optimal_value

        initial_values = np.fromiter(molecule_params.dict.values(), dtype=float)
        method = 'Nelder-Mead'
        options = {'maxiter': cpu_iter}
        return optimize.minimize(fun=optimize_func, x0=initial_values, method=method, options=options)

    @staticmethod
    def get_semi_optimized_ground_state(w: IWaveFunction, qpu_iter: int) -> List:
        param_space = np.linspace(0.1, 3.0, 30)
        print(param_space)
        molecule_params = w.molecule_parameters
        result = list()  # ITypedList(allowed_types=IContainer)
        for par in param_space:
            for i, key in enumerate(molecule_params):
                molecule_params.dict[key] = round(par, 1)  # Temporary rounding to use correct mol_data
            w.update_molecule(molecule_params)
            trial_result = CPU.get_optimized_state(w=w, max_iter=qpu_iter)  # Calculate trial result
            container = IContainer(molecule_params, trial_result.optimal_value, w.molecule.fci_energy)
            result.append(container)
        return result

    @staticmethod
    def get_optimized_state(w: IWaveFunction, max_iter: int) -> OptimizationTrialResult:
        initial_state_circuit = QPU.get_initial_state_circuit(w=w)
        study = QPU.get_variational_study(w=w, p_c=initial_state_circuit, name='HydrogenStudy')
        return QPU.get_optimized_state(s=study, max_iter=max_iter)
