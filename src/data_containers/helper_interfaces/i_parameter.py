from typing import (Dict)
import cirq
# from openfermioncirq.optimization import OptimizationTrialResult


class IParameter:

    def get_resolved(self) -> cirq.study.resolver:
        return cirq.ParamResolver(self._p)

    def get_dict(self) -> Dict[str, float]:
        return self._p

    dict = property(get_dict)

    def __init__(self, parameters: Dict[str, float]):
        self._p = parameters

    # def update(self, r: OptimizationTrialResult):
    #     if len(r.optimal_parameters) != len(self.dict):
    #         raise IndexError("Dimensions of optimized trial result does not correspond to IParameter")
    #     for i, key in enumerate(self.dict.keys()):
    #         self[key] = r.optimal_parameters[i]

    def update(self, v: [float]):
        if len(v) != len(self.dict):
            raise IndexError("Dimensions of optimized trial result does not correspond to IParameter")
        for i, key in enumerate(self.dict.keys()):  # Dirty set parameter values
            self[key] = v[i]

    def __iter__(self):
        return self.dict.__iter__()

    def __str__(self):
        return self.dict.__str__()

    def __getitem__(self, item: str):
        return self.dict.__getitem__(item)

    def __setitem__(self, key: str, value: float):
        return self.dict.__setitem__(key, value)

    def __add__(self, other):
        return IParameter({**self.dict, **other.dict})
