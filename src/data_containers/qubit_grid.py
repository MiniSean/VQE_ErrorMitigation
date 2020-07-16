import cirq


def define_square_grid(diameter: int) -> [cirq.GridQubit]:
    """
    Generates a grid of qubits based on the dimensions provided.
    :param diameter: Square grid diameter
    :return: Array of grid qubits
    """
    return cirq.GridQubit.square(diameter)
