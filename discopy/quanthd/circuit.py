"""
Implementation of the high-dimensional quantum circuits.
"""

from discopy import messages, monoidal, rigid, tensor
from discopy.cat import AxiomError
from discopy.rigid import Ob, Ty, Diagram
from discopy.tensor import np, Dim, Tensor


def _box_type(t, *, exp_size=None, min_dim=2):
    n = 1 if not exp_size else exp_size
    t = Qudit(*(t, )*n) if isinstance(t, int) else t
    if not isinstance(t, Qudit):
        raise TypeError(messages.type_err(Qudit, type_))
    if exp_size and len(t)!=exp_size:
        raise ValueError(f'Expected {exp_size} qudits in {t}, found {len(t)}')
    if min_dim and np.any(np.array(t, dtype=np.int) < min_dim):
        raise ValueError(f'Dimension less than the expected {min_dim}')
    return t


class Qudit(Dim):
    @staticmethod
    def upgrade(old):
        return Qudit(*[x.name for x in old.objects])

    def __repr__(self):
        return "Qudit({})".format(', '.join(map(repr, self)) or '1')


ScalarType = Qudit(1)


@monoidal.Diagram.subclass
class Circuit(tensor.Diagram):
    def __repr__(self):
        return super().__repr__().replace('Diagram', 'HDCircuit')

    def grad(self, var):
        return super().grad(var)
    
    @staticmethod
    def cups(left, right):
        from discopy.quanthd import nadd, H, Bra

        def cup_factory(left, right):
            if left != right or not isinstance(left, Qudit):
                raise ValueError()
            d = left[0]
            return nadd(d) >> H(d) @ Id(d) >> Bra(0, 0, dom=Qudit(d, d)) # TODO mul by sqrt(d)
        return rigid.cups(left, right, ar_factory=Circuit, cup_factory=cup_factory)

    @staticmethod
    def caps(left, right):
        return Circuit.cups(left, right).dagger()


class Id(rigid.Id, Circuit):
    def __init__(self, dom):
        dom = _box_type(dom)
        rigid.Id.__init__(self, dom)
        Circuit.__init__(self, dom, dom, [], [])

    def __repr__(self):
        return "Id({})".format(len(self.dom))   # TODO Review

    def __str__(self):
        return repr(self)


Circuit.id = Id


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
