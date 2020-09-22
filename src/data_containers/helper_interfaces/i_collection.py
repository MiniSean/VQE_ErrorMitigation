# Holds data for every iteration of optimization
from cirq import Gate
from typing import List, Iterable, Iterator
from statistics import stdev, mean
from openfermion import MolecularData
from src.data_containers.helper_interfaces.i_wave_function import IWaveFunction
from src.data_containers.helper_interfaces.i_noise_wrapper import INoiseWrapper
from src.data_containers.helper_interfaces.i_parameter import IParameter


class IContainer:

    def get_molecule_parameter(self) -> float:  # IParameter
        return self._molecular_parameter

    def set_molecule_parameter(self, m_param: float):  # IParameter
        self._molecular_parameter = m_param

    def get_measured_eigenvalue(self) -> float:
        return mean(self._optimized_exp_values)

    def get_measured_standard_deviation(self) -> float:
        return stdev(self._optimized_exp_values)

    def get_fci_value(self) -> float:
        return self._fci_value if self._fci_value is not None else float('NaN')

    def get_hf_value(self) -> float:
        if hasattr(self, '_hf_value'):
            return self._hf_value if self._hf_value is not None else float('NaN')
        else:
            float('NaN')

    molecule_param = property(get_molecule_parameter)  # , set_molecule_parameter)
    measured_value = property(get_measured_eigenvalue)  # , set_measured_eigenvalue)
    measured_std = property(get_measured_standard_deviation)  # , set_measured_eigenvalue)
    fci_value = property(get_fci_value)
    hf_value = property(get_hf_value)

    def __init__(self, m_param: IParameter, e_values: [float], m_data: MolecularData):
        self._molecular_parameter = m_param['r0']
        print(self._molecular_parameter)
        # e_value: float, std_value: float
        self._optimized_exp_values = e_values  # Store all information
        # self._measured_eigenvalue = mean(self._optimized_exp_values)  # e_value
        # self._measured_standard_deviation = stdev(self._optimized_exp_values)  # std_value
        # fci_value: float, hf_value: float
        self._fci_value = m_data.fci_energy  # fci_value
        self._hf_value = m_data.hf_energy  # hf_value


class IMeasurementCollection:

    def get_wave_functions(self) -> Iterator[IWaveFunction]:
        """
        Creates generator which yields an IWaveFunction class based on the initial wave function
        with a specific combination of parameter and noise model setting.
        :return: Generator IWaveFunction depending on specific parameter and noise model setting
        """
        wave_func = self._wave_function_ID  # Define wave function to use
        # Create generative function call
        for par_wave_func in self._get_param_iter(wave_func):
            if len(self.noise_model_space) == 0:
                yield par_wave_func
            else:
                for channel in self.noise_model_space:
                    yield INoiseWrapper(par_wave_func, channel)

    def _get_param_iter(self, wave_func: IWaveFunction) -> Iterator[IWaveFunction]:
        param_layout = wave_func.molecule_parameters
        for par in self.parameter_space:
            for i, key in enumerate(param_layout):
                param_layout.dict[key] = par
            wave_func.update_molecule(param_layout)
            yield wave_func

    def __init__(self, w: IWaveFunction, p_space: List[float], n_space: List[List[Gate]]):
        """
        Measurement collection constructor
        :param w: Wave function to be measured
        :param p_space: (Moleculardata) Parameters to be evaluated
        :param n_space: Noise channels to be evaluated
        """
        # Simple list container
        self._container = list()
        # Measurement specifications
        self._wave_function_ID = w
        self.parameter_space = p_space
        self.noise_model_space = n_space

    def append(self, item: IContainer):
        self._container.append(item)

    def set(self, item_list: List[IContainer]):
        self._container = item_list

    def __iter__(self):
        return self._container.__iter__()
