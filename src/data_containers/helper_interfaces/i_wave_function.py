from abc import abstractmethod


class IWaveFunction:

    @abstractmethod
    def sqrt_probability(self):
        """
        Lambda-func of Wave function expectation value
        :return: <phi|phi>
        """
        raise NotImplemented
