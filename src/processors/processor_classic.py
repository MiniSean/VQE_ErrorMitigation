import numpy as np
import cirq
import copy
from scipy import optimize
from openfermioncirq.optimization import OptimizationTrialResult
from typing import List

from src.processors.processor_quantum import QPU
from src.data_containers.helper_interfaces.i_wave_function import IWaveFunction
from src.data_containers.helper_interfaces.i_noise_wrapper import INoiseWrapper
from src.data_containers.helper_interfaces.i_parameter import IParameter
from src.data_containers.helper_interfaces.i_collection import IContainer, IMeasurementCollection


class CPU:

    @staticmethod
    def get_collection_ground_states(collection: IMeasurementCollection, cpu_iter: int, qpu_iter: int) -> List[IContainer]:
        """
        Iterates through all wave functions to be 'measured' from collection.
        Calculates optimized expectation value through either VariationalStudy (for noiseless wave functions)
        or a custom optimization loop (for noisy wave functions).
        The calculation of expectation values is repeated #cpu_iter number of times where each calculation has a max iteration of #qpu_iter.
        The resulting information is stored in IContainer classes and returned as in a list.
        :param collection: Collection that determines the wave functions to be calculated
        :param cpu_iter: Number of iterations for calculating the mean and std of expectation value results.
        :param qpu_iter: Number of iterations for calculating the expectation value itself.
        :return: List of IContainer classes.
        """
        result = list()
        for wave_function, description in collection.get_wave_functions():
            opt_value_list = []
            for j in range(cpu_iter):
                if isinstance(wave_function, INoiseWrapper):
                    opt_value, par = CPU.get_custom_optimized_state(n_w=copy.deepcopy(wave_function), max_iter=qpu_iter)  # Mixed
                    print(f'par: {par}, val: {opt_value}')
                else:
                    opt_value = CPU.get_optimized_state(w=wave_function, max_iter=qpu_iter).optimal_value  # Ideal
                opt_value_list.append(opt_value)

            container = IContainer(m_param=wave_function.molecule_parameters, e_values=opt_value_list, m_data=wave_function.molecule, label=description)
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
        operator_params = n_w.operator_parameters  # Circuit parameters to optimize
        # Prepare evaluation circuit (With noise)
        evaluation_circuit = n_w.get_noisy_circuit()  # Initial state + Ansatz operators
        # for q in n_w.qubits:
        #     evaluation_circuit.append(cirq.measure(q))  # Measurement

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
        optimize_result = optimize.minimize(fun=optimize_func, x0=initial_values, method=method, options=options, tol=1e-4)
        return optimize_result.fun, optimize_result.x
