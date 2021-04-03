"""
Implementation of the high-dimensional quantum circuits.
"""

from discopy import messages, monoidal, rigid, tensor
from discopy.cat import AxiomError
from discopy.rigid import Ob, Diagram   # TODO Review here
from discopy.tensor import Dim, Tensor
from discopy.quantum import Qudit, qudit, qubit
from discopy.rigid import Ty    # TODO Check quantum.Ty
import numpy as np


def _get_qudit(obj):
    if hasattr(obj, 'objects'):
        obj = obj.objects
        if len(obj) != 1:
            raise ValueError()  # TODO Error spec
        obj = obj[0]
    if not isinstance(obj, Qudit):
        raise TypeError(messages.type_err(Qudit, obj))
    return obj


def _get_qudit_dims(obj):
    if hasattr(obj, 'objects'):
        if not all(map(lambda v: isinstance(v, Qudit), obj.objects)):
            raise TypeError(messages.type_err(Qudit, None))
        return tuple(map(lambda v: v.dim, obj.objects))
    raise TypeError(messages.type_err(type(qudit(2)), obj))


def _box_type(t, *, exp_size=None, min_dim=2):
    n = 1 if not exp_size else exp_size
    t = qudit(t) ** n if isinstance(t, int) else t
    _get_qudit_dims(t)
    if exp_size and len(t) != exp_size:
        raise ValueError(f'Expected {exp_size} qudits in {t}, found {len(t)}')
    if min_dim and np.any(np.array(list(map(lambda obj: obj.dim, t))) < min_dim):
        raise ValueError(f'Dimension less than the expected {min_dim}')
    return t


ScalarType = qubit**0   # TODO Use a common type


@monoidal.Diagram.subclass
class Circuit(tensor.Diagram):
    def __init__(self, dom, cod, *args, **kwargs):
        tensor.Diagram.__init__(self, dom, dom, *args, **kwargs)
        self.id = lambda : Id(dom)

    def __repr__(self):
        return super().__repr__().replace('Diagram', 'HDCircuit')

    def grad(self, var):
        return super().grad(var)
    
    @staticmethod
    def cups(left, right):
        from discopy.quanthd import nadd, H, Bra, Scalar

        def cup_factory(left, right):
            left, right = map(_get_qudit, (left, right))
            if left != right or not isinstance(left, Qudit):
                raise ValueError()
            d = left.dim
            return (H(d) @ Id(d) >> nadd(d)).dagger() \
                >> Bra(0, 0, dom=Ty(left, left)) @ Scalar(d**.5)
        return rigid.cups(Ty(left), Ty(right), ar_factory=Circuit, cup_factory=cup_factory)

    @staticmethod
    def caps(left, right):
        return Circuit.cups(left, right).dagger()
    
    def eval(self):
        functor = tensor.Functor(lambda x: x[0].dim, lambda f: f.array)
        return functor(self)


class Id(rigid.Id, Circuit):
    def __init__(self, dom):
        dom = _box_type(dom)
        rigid.Id.__init__(self, dom)
        Circuit.__init__(self, dom, dom, [], [])

    def __repr__(self):
        return "Id({})".format(len(self.dom))   # TODO Review

    def __str__(self):
        return repr(self)


# Circuit.id = Id


class Box(rigid.Box, Circuit):
    def __init__(self, name, dom, cod, data=None, _dagger=False):
        dom, cod = _box_type(dom), _box_type(cod)
        rigid.Box.__init__(self, name, dom, cod, data=data, _dagger=_dagger)
        Circuit.__init__(self, dom, cod, [self], [0])

    def grad(self, var):
        if var not in self.free_symbols:
            return Sum([], self.dom, self.cod)
        raise NotImplementedError

    def __repr__(self):
        return self.name


class Sum(monoidal.Sum, Box):
    @staticmethod
    def upgrade(old):
        return Sum(old.terms, old.dom, old.cod)

    def grad(self, var):
        return sum(circuit.grad(var) for circuit in self.terms)


Circuit.sum = Sum
