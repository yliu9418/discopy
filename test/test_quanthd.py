from pytest import raises
from discopy.quanthd import *
from discopy.quanthd.gates import SINGLE_QUDIT_GATE_CLASSES
import discopy.quantum.gates as qugates
import numpy as np
from functools import reduce


def _assert_norm_0(m):
    assert np.isclose(np.linalg.norm(m), 0.)


def _assert_op_is_iden(op):
    m = op.eval().array
    m = np.asarray(m)
    assert np.isclose(np.linalg.norm(m - np.eye(*m.shape[:2])), 0.)


def _assert_eval_op_diff_0(op1, op2):
    _assert_norm_0(op1.eval().array - op2.eval().array)


def test_dim_2():
    qubit_id_op = qugates.Box.id(qugates.qubit)
    equiv_pairs = [(H(2), qugates.H), (X(2), qugates.X),
                   (Z(2), qugates.Z), (Neg(2), qubit_id_op)]
    for pair in equiv_pairs:
        _assert_eval_op_diff_0(pair[0], pair[1])


def _op_pow(op, n):
    return reduce(lambda a, b: a >> b, [op]*n, Box.id(op.dom))


def test_basic_identities():
    for d in range(2, 9):
        _assert_op_is_iden(_op_pow(H(d), 4))    # H**4=I
        _assert_op_is_iden(_op_pow(X(d), d))    # X**d=I
        _assert_op_is_iden(_op_pow(Z(d), d))    # Z**d=I
        _assert_op_is_iden(_op_pow(Neg(d), 2))  # Neg**2=I
        # HXH^H=Z
        _assert_op_is_iden(H(d) >> X(d) >> H(d).dagger() >> Z(d).dagger())
        # HZH^H=X
        # _assert_op_is_iden(H(d) >> Z(d) >> H(d).dagger() >> X(d).dagger())

    # H is Hadamard when d is 2, thus H**2=I
    _assert_op_is_iden(_op_pow(H(2), 2))

    from itertools import product
    for op, d in product(SINGLE_QUDIT_GATE_CLASSES, range(2, 9)):
        _assert_op_is_iden(op(d) >> op(d).dagger())


def test_braket():
    from itertools import product
    for d in range(2, 5):
        for i, j in product(range(d), repeat=2):
            # <i|j>=delta_{i, j}
            m = (Ket(i, cod=d) >> Bra(j, dom=d)).eval().array
            _assert_norm_0(float(m) - (i==j))

            m = (Bra(i, dom=d).dagger() >> Ket(j, cod=d).dagger()).eval().array
            _assert_norm_0(float(m) - (i==j))

    for d in range(2, 9):
        m = (Ket(0, cod=d) >> H(d)).eval().array
        _assert_norm_0(m*(d/np.sqrt(d)) - np.ones(d))
