import numpy as np
import cirq
from statistics import stdev, mean
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
    def get_specific_ground_states(p_space: np.ndarray, w: IWaveFunction, cpu_iter: int, qpu_iter: int) -> List:
        """
        Uses pre-calculated data points to update molecule geometry.
        Calculates optimized expectation value through either VariationalStudy (for noiseless wave functions)
        or a custom optimization loop (for noisy wave functions).
        The calculation of expectation values is repeated #cpu_iter number of times where each calculation has a max iteration of #qpu_iter.
        The resulting information is stored in IContainer classes and returned as in a list.
        :param p_space: Linear space of molecule parameters being evaluated
        :param w: (Noisy) IWaveFunction class that determines the calculation method for expectation value.
        :param cpu_iter: Number of iterations for calculating the mean and std of expectation value results.
        :param qpu_iter: Number of iterations for calculating the expectation value itself.
        :return: List of IContainer classes.
        """
        # p_space = np.linspace(0.1, 3.0, 30)
        molecule_params = w.molecule_parameters
        result = list()  # ITypedList(allowed_types=IContainer)
        for par in p_space:
            for i, key in enumerate(molecule_params):
                molecule_params.dict[key] = par  # Temporary rounding to use correct mol_data
            w.update_molecule(molecule_params)

            # Implement statistical average and standard deviation
            opt_value_list = []
            for j in range(cpu_iter):
                if isinstance(w, INoiseWrapper):
                    opt_value = CPU.get_custom_optimized_state(n_w=w, max_iter=qpu_iter)[0]
                else:
                    opt_value = CPU.get_optimized_state(w=w, max_iter=qpu_iter).optimal_value  # Calculate trial result
                opt_value_list.append(opt_value)
            avg_value = mean(opt_value_list)
            std_value = stdev(opt_value_list)

            container = IContainer(m_param=molecule_params, e_value=avg_value, std_value=std_value, fci_value=w.molecule.fci_energy, hf_value=w.molecule.hf_energy)
            result.append(container)
        return result

    @staticmethod
    def get_optimized_state(w: IWaveFunction, max_iter: int) -> OptimizationTrialResult:
        initial_state_circuit = QPU.get_initial_state_circuit(w=w)
        study = QPU.get_variational_study(w=w, p_c=initial_state_circuit, name='HydrogenStudy')
        return QPU.get_optimized_state(s=study, max_iter=max_iter)

    @staticmethod
    def get_custom_optimized_state(n_w: INoiseWrapper, max_iter: int) -> (float, List[float]):
        """
        Performs custom operator parameter optimization.
        A quantum circuit is created with initial state preparation and ansatz operators.
        This circuit depends on sympy.Symbol variables which are subject to optimization.
        :param n_w: Stationary NoiseWrapper(IWaveFunction) to evaluate expectation value from
        :param noise_model: Noise model needed for the expectation value
        :param max_iter: Maximum iteration steps for (classic) scipy optimize
        :return:
        """
        operator_params = n_w.operator_parameters
        # Prepare evaluation circuit (With noise
        evaluation_circuit = n_w.get_noisy_circuit()  # Initial state + Ansatz operators

        def update_variational_parameters(p: IParameter, v: np.ndarray) -> IParameter:
            for i, key in enumerate(p):
                p.dict[key] = v[i]
            return p

        def evaluate_circuit(c: cirq.circuits.circuit, p: IParameter) -> float:
            exp_value = QPU.get_simulated_noisy_expectation_value(w=n_w, r_c=c, r=p.get_resolved())
            return exp_value

        def optimize_func(x: np.ndarray) -> float:
            update_variational_parameters(p=operator_params, v=x)
            result = evaluate_circuit(c=evaluation_circuit, p=operator_params)
            return result

        initial_values = np.fromiter(operator_params.dict.values(), dtype=float)
        method = 'Nelder-Mead'
        options = {'maxiter': max_iter}
        optimize_result = optimize.minimize(fun=optimize_func, x0=initial_values, method=method, options=options)
        return optimize_result.fun, optimize_result.x
