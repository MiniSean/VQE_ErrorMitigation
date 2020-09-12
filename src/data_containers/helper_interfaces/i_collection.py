# Holds data for every iteration of optimization
from openfermioncirq.optimization import OptimizationTrialResult
from src.data_containers.helper_interfaces.i_parameter import IParameter


class IContainer(object):

    def get_molecule_parameter(self) -> float:  # IParameter
        return self._molecular_parameter

    def set_molecule_parameter(self, m_param: float):  # IParameter
        self._molecular_parameter = m_param

    def get_measured_eigenvalue(self) -> float:
        return self._measured_eigenvalue

    def set_measured_eigenvalue(self, e_value: float):
        self._measured_eigenvalue = e_value

    def get_fci_value(self) -> float:
        return self._fci_value

    molecule_param = property(get_molecule_parameter)  # , set_molecule_parameter)
    measured_value = property(get_measured_eigenvalue)  # , set_measured_eigenvalue)
    fci_value = property(get_fci_value)

    def __init__(self, m_param: IParameter, e_value: float, fci_value: float):
        self._molecular_parameter = m_param['r0']
        print(self._molecular_parameter)
        self._measured_eigenvalue = e_value
        self._fci_value = fci_value

    # @staticmethod
    # def json_hook(obj_to_decode):
    #     if '_molecular_parameter' in obj_to_decode and '_measured_eigenvalue' in obj_to_decode and '_fci_value' in obj_to_decode:
    #         return IContainer(obj_to_decode['_molecular_parameter'], obj_to_decode['_measured_eigenvalue'], obj_to_decode['_fci_value'])
    #     else:
    #         return obj_to_decode
    #
    # def to_json(self):
    #     return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)
