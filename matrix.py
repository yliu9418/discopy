""" Implements dagger monoidal functors into matrices.

>>> n = Ty('n')
>>> Alice = Box('Alice', Ty(), n)
>>> loves = Box('loves', n, n)
>>> Bob = Box('Bob', n, Ty())
>>> ob, ar = {n: Dim(2)}, {Alice: [0, 1], loves: [0, 1, 1, 0], Bob: [1, 0]}
>>> F = MatrixFunctor(ob, ar)
>>> c = F(Alice >> loves >> Bob)
>>> c
Matrix(dom=Dim(1), cod=Dim(1), array=[1])
"""

import numpy as np
from functools import reduce as fold
from discopy import cat
from discopy.moncat import Ob, Ty, Box, Diagram, MonoidalFunctor


class Dim(Ty):
    """ Implements dimensions as tuples of positive integers.
    Dimensions form a monoid with product @ and unit Dim(1).

    >>> Dim(1) @ Dim(2) @ Dim(3)
    Dim(2, 3)
    """
    def __init__(self, *xs):
        """
        >>> len(Dim(2, 3, 4))
        3
        >>> len(Dim(1, 2, 3))
        2
        >>> Dim(-1)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: Expected positive integer, got -1 instead.
        """
        for x in xs:
            if not isinstance(x, int) or x < 1:
                raise ValueError("Expected positive integer, got {} instead."
                                 .format(repr(x)))
        super().__init__(*[Ob(x) for x in xs if x > 1])

    def __add__(self, other):
        """
        >>> assert Dim(1) + Dim(2, 3) == Dim(2, 3) + Dim(1) == Dim(2, 3)
        """
        return Dim(*[x.name for x in super().__add__(other)])

    def __getitem__(self, key):
        """
        >>> assert Dim(2, 3)[:1] == Dim(3, 2)[1:] == Dim(2)
        >>> assert Dim(2, 3)[0] == Dim(3, 2)[1] == 2
        """
        if isinstance(key, slice):
            return Dim(*[x.name for x in super().__getitem__(key)])
        return super().__getitem__(key).name

    def __iter__(self):
        """
        >>> [n for n in Dim(2, 3, 4)]
        [2, 3, 4]
        """
        for i in range(len(self)):
            yield self[i]

    def __repr__(self):
        """
        >>> Dim(1, 2, 3)
        Dim(2, 3)
        """
        return "Dim({})".format(', '.join(map(repr, self)) or '1')

    def __str__(self):
        """
        >>> print(Dim(1, 2, 3))
        Dim(2, 3)
        """
        return repr(self)

    def __hash__(self):
        """
        >>> dim = Dim(2, 3)
        >>> {dim: 42}[dim]
        42
        """
        return hash(repr(self))

class Matrix(Diagram):
    """ Implements a matrix with dom, cod and numpy array.

    >>> m = Matrix(Dim(2), Dim(2), [0, 1, 1, 0])
    >>> v = Matrix(Dim(1), Dim(2), [0, 1])
    >>> v >> m >> v.dagger()
    Matrix(dom=Dim(1), cod=Dim(1), array=[0])
    """
    def __init__(self, dom, cod, array):
        """
        >>> Matrix(Dim(2), Dim(2), [0, 1, 1, 0])
        Matrix(dom=Dim(2), cod=Dim(2), array=[0, 1, 1, 0])
        """
        dom, cod = Dim(*dom), Dim(*cod)
        array = np.array(array).reshape(dom + cod)
        self._dom, self._cod, self._array = dom, cod, array

    @property
    def array(self):
        """
        >>> Matrix(Dim(2, 2), Dim(2), [1, 0, 0, 1, 0, 1, 1, 0]).array.shape
        (2, 2, 2)
        >>> Matrix(Dim(2, 2), Dim(2), [1, 0, 0, 1, 0, 1, 1, 0]).array.flatten()
        array([1, 0, 0, 1, 0, 1, 1, 0])
        """
        return self._array

    def __repr__(self):
        """
        >>> Matrix(Dim(2, 2), Dim(2), [1, 0, 0, 1, 0, 1, 1, 0])
        Matrix(dom=Dim(2, 2), cod=Dim(2), array=[1, 0, 0, 1, 0, 1, 1, 0])
        """
        return "Matrix(dom={}, cod={}, array={})".format(
                       self.dom, self.cod, list(self.array.flatten()))

    def __str__(self):
        """
        >>> print(Matrix(Dim(2, 2), Dim(2), [1, 0, 0, 1, 0, 1, 1, 0]))
        Matrix(dom=Dim(2, 2), cod=Dim(2), array=[1, 0, 0, 1, 0, 1, 1, 0])
        """
        return repr(self)

    def __eq__(self, other):
        """
        >>> arr = np.array([1, 0, 0, 1, 0, 1, 1, 0]).reshape((2, 2, 2))
        >>> m = Matrix(Dim(2, 2), Dim(2), arr)
        >>> assert m == m and np.all(m == arr)
        """
        if not isinstance(other, Matrix):
            return self.array == other
        return (self.dom, self.cod) == (other.dom, other.cod) and np.all(
                self.array == other.array)

    def then(self, other):
        """
        >>> m = Matrix(Dim(2), Dim(2), [0, 1, 1, 0])
        >>> assert m >> m == m >> m.dagger() == Id(2)
        """
        if self.cod != other.dom:
            raise AxiomError("{} does not compose with {}."
                                   .format(repr(self), repr(other)))
        return Matrix(self.dom, other.cod,
                      np.tensordot(self.array, other.array, len(self.cod)))

    def tensor(self, other):
        """
        >>> v = Matrix(Dim(1), Dim(2), [1, 0])
        >>> v @ v
        Matrix(dom=Dim(1), cod=Dim(2, 2), array=[1, 0, 0, 0])
        >>> v @ v.dagger()
        Matrix(dom=Dim(2), cod=Dim(2), array=[1, 0, 0, 0])
        >>> assert v @ v.dagger() == v << v.dagger()
        """
        dom, cod = self.dom + other.dom, self.cod + other.cod
        array = np.tensordot(self.array, other.array, 0)
        return Matrix(dom, cod, array)

    def dagger(self):
        """
        >>> m = Matrix(Dim(2, 2), Dim(2), [1, 0, 0, 1, 0, 1, 1, 0])
        >>> m.dagger()
        Matrix(dom=Dim(2), cod=Dim(2, 2), array=[1, 0, 0, 1, 0, 1, 1, 0])
        >>> v = Matrix(Dim(1), Dim(2), [0, 1])
        >>> assert (v >> m.dagger()).dagger() == m >> v.dagger()
        """
        array = np.moveaxis(self.array, range(len(self.dom + self.cod)),
            [i + len(self.cod) if i < len(self.dom) else
             i - len(self.dom) for i in range(len(self.dom + self.cod))])
        return Matrix(self.cod, self.dom, np.conjugate(array))

    @staticmethod
    def id(dim):
        """
        >>> assert Id(2) == Matrix.id(2) == Matrix(Dim(2), Dim(2), [1, 0, 0, 1])
        """
        return Id(dim)

class AxiomError(cat.AxiomError):
    """
    >>> m = Matrix(Dim(2, 2), Dim(2), [1, 0, 0, 1, 0, 1, 1, 0])
    >>> m >> m  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    matrix.AxiomError: Matrix... does not compose with Matrix...
    """
    pass

class Id(Matrix):
    """ Implements the identity matrix for a given dimension.

    >>> Id(1)
    Matrix(dom=Dim(1), cod=Dim(1), array=[1])
    >>> Id(2)
    Matrix(dom=Dim(2), cod=Dim(2), array=[1.0, 0.0, 0.0, 1.0])
    >>> Id(1, 2, 3)  # doctest: +ELLIPSIS
    Matrix(dom=Dim(2, 3), cod=Dim(2, 3), array=[1.0, ..., 1.0])
    """
    def __init__(self, *dim):
        """
        >>> Id(2).array
        array([[1., 0.],
               [0., 1.]])
        >>> Id(2).array.shape
        (2, 2)
        """
        dim = dim[0] if isinstance(dim[0], Dim) else Dim(*dim)
        array = np.moveaxis(
            fold(lambda a, x: np.tensordot(a, np.identity(x), 0), dim, 1),
            [2 * i for i in range(len(dim))], [i for i in range(len(dim))])
        super().__init__(dim, dim, array)

class MatrixFunctor(MonoidalFunctor):
    """ Implements a matrix-valued functor.

    >>> x, y = Ty('x'), Ty('y')
    >>> f = Box('f', x, x @ y)
    >>> F = MatrixFunctor({x: 1, y: 2}, {f: [0, 1]})
    >>> F(f)
    Matrix(dom=Dim(1), cod=Dim(2), array=[0, 1])
    """
    def __init__(self, ob, ar):
        """
        >>> MatrixFunctor({Ty('x'): 2}, {})
        MatrixFunctor(ob={Ty('x'): Dim(2)}, ar={})
        >>> MatrixFunctor({Ty('x'): Dim(2)}, {})
        MatrixFunctor(ob={Ty('x'): Dim(2)}, ar={})
        >>> MatrixFunctor({Ty('x'): Ty('y')}, {})  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: Expected int or Dim object, got Ty('y') instead.
        """
        for x, y in ob.items():
            if isinstance(y, int):
                ob.update({x: Dim(y)})
            elif not isinstance(y, Dim):
                raise ValueError(
                    "Expected int or Dim object, got {} instead."
                    .format(repr(y)))
        super().__init__(ob, ar)

    def __repr__(self):
        """
        >>> x, y = Ty('x'), Ty('y')
        >>> MatrixFunctor({x: 1, y: 3}, {})
        MatrixFunctor(ob={Ty('x'): Dim(1), Ty('y'): Dim(3)}, ar={})
        >>> MatrixFunctor({}, {Box('f', x @ x, y): list(range(3))})
        MatrixFunctor(ob={}, ar={Box('f', Ty('x', 'x'), Ty('y')): [0, 1, 2]})
        """
        return super().__repr__().replace("MonoidalFunctor", "MatrixFunctor")

    def __call__(self, d):
        """
        >>> x, y = Ty('x'), Ty('y')
        >>> f, g = Box('f', x @ x, y), Box('g', y, Ty())
        >>> ob = {x: 2, y: 3}
        >>> ar = {f: list(range(2 * 2 * 3)), g: list(range(3))}
        >>> F = MatrixFunctor(ob, ar)
        >>> F(f >> g).array.flatten()
        array([ 5., 14., 23., 32.])
        >>> F(f @ f.dagger()).array.shape
        (2, 2, 3, 3, 2, 2)
        >>> F(f.dagger() @ f).array.shape
        (3, 2, 2, 2, 2, 3)
        >>> F(f.dagger() >> f).array.flatten()
        array([126., 144., 162., 144., 166., 188., 162., 188., 214.])
        >>> F(g.dagger() >> g).array
        array(5)
        """
        if isinstance(d, Ty):
            return sum([self.ob[x] for x in d], Dim(1))
        elif isinstance(d, Box):
            if d._dagger: return Matrix(
                self(d.cod), self(d.dom), self.ar[d.dagger()]).dagger()
            return Matrix(self(d.dom), self(d.cod), self.ar[d])
        scan, array, dim = d.dom, Id(self(d.dom)).array, lambda t: len(self(t))
        for f, offset in d:
            n = dim(scan[:offset])
            source = range(dim(d.dom) + n, dim(d.dom) + n + dim(f.dom))
            target = range(dim(f.dom))
            array = np.tensordot(array, self(f).array, (source, target))
            source = range(len(array.shape) - dim(f.cod), len(array.shape))
            target = range(dim(d.dom) + n, dim(d.dom) + n + dim(f.cod))
            array = np.moveaxis(array, source, target)
            scan = scan[:offset] + f.cod + scan[offset + len(f.dom):]
        return Matrix(self(d.dom), self(d.cod), array)
