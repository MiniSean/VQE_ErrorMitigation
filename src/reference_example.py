# Implements VQE example notebook
from typing import Iterable, Sequence
import cirq
import sympy
import openfermioncirq
from openfermioncirq.optimization import COBYLA, NELDER_MEAD, OptimizationParams
from openfermion import MolecularData, jordan_wigner, eigenspectrum, QubitOperator
import numpy as np


class H2Ansatz(openfermioncirq.VariationalAnsatz):

    def __init__(self):
        self._theta = sympy.Symbol('theta')
        self._qubits = [cirq.GridQubit(i, j) for i in range(2) for j in range(2)]
        super().__init__()

    def params(self) -> Iterable[sympy.Symbol]:
        return [self._theta]

    def operations(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        yield [cirq.ry(np.pi).on(qubits[0]),
               cirq.ry(np.pi).on(qubits[1])]
        yield [cirq.rx(np.pi / 2).on(qubits[0]),
               cirq.ry(np.pi / 2).on(qubits[1]),
               cirq.ry(np.pi / 2).on(qubits[2]),
               cirq.ry(np.pi / 2).on(qubits[3])]
        yield [cirq.CNOT(qubits[0], qubits[1]),
               cirq.CNOT(qubits[1], qubits[2]),
               cirq.CNOT(qubits[2], qubits[3])]
        yield cirq.rz(self._theta).on(qubits[3])
        yield [cirq.CNOT(qubits[2], qubits[3]),
               cirq.CNOT(qubits[1], qubits[2]),
               cirq.CNOT(qubits[0], qubits[1])]
        yield [cirq.rx(-np.pi / 2).on(qubits[0]),
               cirq.ry(-np.pi / 2).on(qubits[1]),
               cirq.ry(-np.pi / 2).on(qubits[2]),
               cirq.ry(-np.pi / 2).on(qubits[3])]

    def _generate_qubits(self) -> Sequence[cirq.Qid]:
        return self._qubits


def get_hamiltonian() -> QubitOperator:
    diatomic_bond_length = .7414

    geometry = [('H', (0., 0., 0.)),
                ('H', (0., 0., diatomic_bond_length))]

    basis = 'sto-3g'
    multiplicity = 1
    charge = 0
    description = format(diatomic_bond_length)

    molecule = MolecularData(geometry, basis, multiplicity, description=description)
    molecule.load()

    return jordan_wigner(molecule.get_molecular_hamiltonian())


def get_ansatz() -> H2Ansatz:
    return H2Ansatz()


def get_ansatz_study(h: QubitOperator, w: openfermioncirq.VariationalAnsatz) -> openfermioncirq.VariationalStudy:
    objective = openfermioncirq.HamiltonianObjective(h)
    return openfermioncirq.VariationalStudy(name='UCC_single_term', ansatz=w, objective=objective)


def get_optimization(s: openfermioncirq.VariationalStudy) -> openfermioncirq.optimization.OptimizationTrialResult:
    optimization_params = OptimizationParams(algorithm=COBYLA, initial_guess=[0.01])  # dtype='uint64'
    seed_array = [np.random.randint(2**31) for i in optimization_params.initial_guess]
    return s.optimize(optimization_params, seeds=seed_array)


def get_resolved_circuit(s: openfermioncirq.VariationalStudy, r: openfermioncirq.optimization.OptimizationTrialResult) -> cirq.circuits.circuit:
    resolver = cirq.ParamResolver({'theta': r.optimal_parameters})
    return cirq.resolve_parameters(s.circuit, resolver)


if __name__ == '__main__':
    print(np.random.randint(2**31))

    hamiltonian = get_hamiltonian()
    ansatz = get_ansatz()

    study = get_ansatz_study(hamiltonian, ansatz)
    print(study.circuit)

    result = get_optimization(study)
    print('Optimized VQE result: {}'.format(result.optimal_value))
    print('Target Hamiltonian eigenvalue: {}'.format(eigenspectrum(hamiltonian)[0]))
    print('Optimal angle: {}'.format(result.optimal_parameters))

    resolved_circuit = get_resolved_circuit(study, result)
    print(resolved_circuit)
