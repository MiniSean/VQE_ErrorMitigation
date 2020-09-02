import cirq
import numpy as np
import openfermion as of
from random import uniform
from typing import List


# Allows the modification of QubitOperator lists by introducing arbitrary or white noise-channels
class Noisify:

    @staticmethod
    def introduce_noise(circuit: cirq.Circuit) -> cirq.Circuit:
        """
        Introduces noise operations after each gate operation in a circuit.
        Current noise operations are based on a random rotation to anywhere on the Bloch sphere.
        :param circuit: Collection of gate operation to introduce noise to
        :return: Same circuit with noise operations included
        """
        result = []
        for moment in circuit.moments:
            for gate_op in moment.operations:
                noise_op = Noisify.uniform_rotation_gate(1000)  # Create random noise operation per gate operation
                new_moments = [gate_op]
                for q_op in noise_op:
                    noise_model = cirq.ConstantQubitNoiseModel(q_op)
                    # Handle obscure noisy_operation function output
                    noise_moment = noise_model.noisy_operation(operation=new_moments[0])
                    new_moments.append(noise_moment[1])  # Collects the added noise moment after gate operation
                # Append to new moment
                result.append(new_moments)
        # Create new circuit and set order strategy
        return cirq.Circuit(result, strategy=cirq.InsertStrategy.EARLIEST)

    @staticmethod
    def uniform_rotation_gate(sample_count: int) -> [of.QubitOperator]:
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
