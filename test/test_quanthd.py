from pytest import raises
from discopy.quanthd import *
from discopy.quanthd.gates import SINGLE_QUDIT_GATE_CLASSES
import discopy.quantum.gates as qugates
import numpy as np
import sympy as sy
from functools import reduce
import itertools


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


def _array_to_square_mat(m):
    m = np.asarray(m)
    return m.reshape((int(np.sqrt(m.size)), )*2)


def _op_pow(op, n):
    return reduce(lambda a, b: a >> b, [op]*n, Box.id(op.dom))


def test_dim_2():
    # Special case d=2 (qubit)
    assert X(2).dom == Qudit(2)
    assert Add(2).dom == Qudit(2, 2)

    qubit_id_op = qugates.Box.id(qugates.qubit)
    equiv_pairs = [(H(2), qugates.H), (X(2), qugates.X),
                   (Z(2), qugates.Z), (Neg(2), qubit_id_op),
                   (Add(2), qugates.CX), (nadd(2), qugates.CX)]
    for pair in equiv_pairs:
        _assert_eval_op_diff_0(pair[0], pair[1])


def test_basic_identities():
    for d in range(2, 9):
        _assert_op_is_iden(_op_pow(H(d), 4))    # H**4=I
        _assert_op_is_iden(_op_pow(X(d), d))    # X**d=I
        _assert_op_is_iden(_op_pow(Z(d), d))    # Z**d=I
        _assert_op_is_iden(_op_pow(Neg(d), 2))  # Neg**2=I
        # Similarity of operators X and Z: HXH^H=Z
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
            assert Ket(i, cod=d).dom == Qudit(d)**0
            assert Ket(i, cod=d).cod == Qudit(d)
            # <i|j>=delta_{i, j}
            m = (Ket(i, cod=d) >> Bra(j, dom=d)).eval().array
            _assert_norm_0(float(m) - (i==j))

            assert Bra(i, dom=d).dom == Qudit(d)
            assert Bra(i, dom=d).cod == Qudit(d)**0
            m = (Bra(i, dom=d).dagger() >> Ket(j, cod=d).dagger()).eval().array
            _assert_norm_0(float(m) - (i==j))

            assert Ket(i, cod=d).dagger() == Bra(i, dom=d)
            assert Ket(i, cod=d) == Bra(i, dom=d).dagger()

    for d in range(2, 9):
        m = (Ket(0, cod=d) >> H(d) @ Scalar(sy.sqrt(d))).eval().array
        _assert_norm_0(m - np.ones(d))


def test_add():
    for dom in [Qudit(2, 3), Qudit(1, 2), 1]:
        with raises(ValueError):
            Add(dom)

    for d in range(2, 5):
        assert Add(d).dom == Add(d).cod == Qudit(d, d)

        m = Add(d).eval().array
        m = _array_to_square_mat(m)
        assert m.shape == (d**2, )*2
        # Operator matrix expected to be a permutation matrix
        # i.e. entries in {0, 1} and doubly stochastic.
        _assert_norm_0(np.unique(m) - np.array([0, 1]))
        _assert_norm_0(m @ np.ones_like(m) - np.ones_like(m))
        _assert_norm_0(np.ones_like(m) @ m - np.ones_like(m))

        # NADD self-inverse
        _assert_op_is_iden(_op_pow(nadd(d), 2), force_square_mat=True)


def test_cups_caps():
    for d in range(2, 7):
        # Circle
        _assert_norm_0(complex((caps(d) >> cups(d)).eval().array) - d)

    for d, k in itertools.product(range(2, 5), range(1, 4)):
        # Multiple circles
        t = Qudit(d)**k
        c = Circuit.caps(t, t) >> Circuit.cups(t, t)
        _assert_norm_0(complex(c.eval().array) - d**k)

    for d in range(2, 7):
        # Yanking equations
        _assert_op_is_iden((Id(d) @ caps(d)) >> (cups(d) @ Id(d)))
        _assert_op_is_iden((caps(d) @ Id(d)) >> (Id(d) @ cups(d)))


def test_trace():
    for d in range(2, 7):
        _assert_norm_0(complex(trace(Id(d)).eval().array) - d)

    for d in range(2, 7):
        # X's matrix is a permutation matrix which has always 0 fixed
        # points => trace(X) = 0
        _assert_norm_0(complex(trace(X(d)).eval().array))
        _assert_norm_0(complex(trace(X(d) @ X(d)).eval().array))


def test_copy_plus():
    for d in range(2, 3):
        # FIXME d > 2
        # Pruning elements for copy dot
        t = Qudit(d)
        sqrt_d_x_0 = Scalar(sy.sqrt(d)) @ Ket(0, cod=t) >> H(t)
        _assert_eval_op_diff_0(sqrt_d_x_0 >> copy(t), caps(t))
        _assert_eval_op_diff_0(copy(t) >> (sqrt_d_x_0.dagger() @ Id(t)), Id(t))
        _assert_eval_op_diff_0(copy(t) >> (Id(t) @ sqrt_d_x_0.dagger()), Id(t))
