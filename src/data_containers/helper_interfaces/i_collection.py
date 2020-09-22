# Holds data for every iteration of optimization
from statistics import stdev, mean
from openfermion import MolecularData
from openfermioncirq.optimization import OptimizationTrialResult
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

