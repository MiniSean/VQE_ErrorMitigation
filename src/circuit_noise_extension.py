import cirq
import numpy as np
import openfermion as of
from random import uniform
from typing import List, Sequence, Callable

from src.data_containers.helper_interfaces.i_parameter import IParameter
from src.data_containers.helper_interfaces.i_wave_function import IWaveFunction


# Allows the modification of QubitOperator lists by introducing arbitrary or white noise-channels
class Noisify:

    @staticmethod
    def introduce_noise_type(op_tree: cirq.OP_TREE, noise_func: Callable[[], List[cirq.Gate]]) -> cirq.OP_TREE:
        result = []
        for op_list in op_tree:
            for gate_op in op_list:
                noise_list = noise_func()  # Create random noise operation per gate operation
                new_gate_list = [gate_op]
                for noise_op in noise_list:
                    noise_model = cirq.ConstantQubitNoiseModel(noise_op)
                    # Check if evaluated gate is no virtual noise moment
                    if isinstance(new_gate_list[0], cirq.Moment) and noise_model.is_virtual_moment(new_gate_list[0]):
                        continue
                    # Handle obscure noisy_operation function output
                    new_moment_list = noise_model.noisy_operation(operation=new_gate_list[0])
                    new_gate_list.append(new_moment_list[1])  # Collects the added noise moment after gate operation
                # Append to new moment
                result.append(new_gate_list)
        result = [val for sublist in result for val in sublist]  # Flatten list of list of cirq.Moments
        return result

    @staticmethod
    def introduce_noise_tree(op_tree: cirq.OP_TREE) -> cirq.OP_TREE:
        noise_func = lambda: Noisify.uniform_rotation_gate(1000)
        result = Noisify.introduce_noise_type(op_tree=op_tree, noise_func=noise_func)
        return result

    @staticmethod
    def uniform_rotation_gate(sample_count: int) -> [cirq.Gate]:  # [of.QubitOperator]
        """
        Mimics the uniform fibonacci mapping on disk but then for a sphere.
        :param sample_count: Number of points to choose from on spherical surface (example: 1000)
        :return: Qubit rotation from |0> to this position on the Bloch sphere
        """
        ratio = 1.6180339887  # Fibonacci ratio approximation
        index = np.random.randint(0, sample_count)  # Uniform random index choice
        phi = np.cos(1 - 2 * index * sample_count)
        theta = 2 * np.pi * ratio * index
        return Noisify.drift_gate(phi, theta)

    @staticmethod
    def orbital_drift_gate() -> [of.QubitOperator]:
        """Simplest way to randomize a rotation. Not uniform"""
        phi = uniform(0, np.pi)
        theta = uniform(0, 2 * np.pi)
        return Noisify.drift_gate(phi, theta)

    @staticmethod
    def drift_gate(phi: float, theta: float) -> [of.QubitOperator]:
        """Translate polar and azimuthal angle rotation to a qubit operator"""
        return [cirq.rx(phi), cirq.rz(theta)]
