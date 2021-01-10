from pytest import raises
from discopy.quanthd import *
from discopy.quanthd.gates import SINGLE_QUDIT_GATE_CLASSES
import discopy.quantum.gates as qugates
import numpy as np
from functools import reduce


def _assert_norm_0(m):
    assert np.isclose(np.linalg.norm(m), 0.)


def _assert_op_is_iden(op, force_square_mat=False):
    m = op.eval().array
    m = np.asarray(m)
    if force_square_mat:
        m = _array_to_square_mat(m)
    assert np.isclose(np.linalg.norm(m - np.eye(*m.shape[:2])), 0.)


def _assert_eval_op_diff_0(op1, op2):
    _assert_norm_0(op1.eval().array - op2.eval().array)


def test_dim_2():
    dom2 = Qudit(2, 2)
    qubit_id_op = qugates.Box.id(qugates.qubit)
    equiv_pairs = [(H(2), qugates.H), (X(2), qugates.X),
                   (Z(2), qugates.Z), (Neg(2), qubit_id_op),
                   (Add(dom2), qugates.CX), (nadd(dom2), qugates.CX)]
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


def _array_to_square_mat(m):
    m = np.asarray(m)
    return m.reshape((int(np.sqrt(m.size)), )*2)


def test_add():
    for dom in [Qudit(2, 3), Qudit(1, 2), 1, 2]:
        with raises(ValueError):
            Add(dom)

    for d in range(2, 5):
        dom2 = Qudit(d, d)
        m = Add(dom2).eval().array
        m = _array_to_square_mat(m)
        assert m.shape == (d**2, )*2
        # Operator matrix expected to be a permutation matrix
        # i.e. entries in {0, 1} and doubly stochastic.
        _assert_norm_0(np.unique(m) - np.array([0, 1]))
        _assert_norm_0(m @ np.ones_like(m) - np.ones_like(m))
        _assert_norm_0(np.ones_like(m) @ m - np.ones_like(m))

        # NADD self-inverse
        _assert_op_is_iden(_op_pow(nadd(dom2), 2), force_square_mat=True)
