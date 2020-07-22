from typing import (Dict)
import cirq


class IParameter:

    def get_resolved(self) -> cirq.study.resolver:
        return cirq.ParamResolver(self._p)

    def get_dict(self) -> Dict[str, float]:
        return self._p

    dict = property(get_dict)

    def __init__(self, parameters: Dict[str, float]):
        self._p = parameters

    def __iter__(self):
        return self.dict.__iter__()

    def __str__(self):
        return self.dict.__str__()

    def __getitem__(self, item: str):
        return self.dict.__getitem__(item)

    def __add__(self, other):
        return IParameter({**self.dict, **other.dict})
