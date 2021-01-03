# -*- coding: utf-8 -*-

from discopy import messages, monoidal, rigid, tensor
from discopy.cat import AxiomError
from discopy.rigid import Ob, Ty, Diagram
from discopy.tensor import np, Dim, Tensor


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


class Id(rigid.Id, Circuit):
    def __init__(self, dom):
        if isinstance(dom, int):
            dom = Qudit(dom)
        rigid.Id.__init__(self, dom)
        Circuit.__init__(self, dom, dom, [], [])

    def __repr__(self):
        return "Id({})".format(len(self.dom))   # TODO Redo

    def __str__(self):
        return repr(self)


Circuit.id = Id


class Box(rigid.Box, Circuit):
    def __init__(self, name, dom, cod,
                 data=None, _dagger=False):
        if dom and not isinstance(dom, Qudit):
            raise TypeError(messages.type_err(Qudit, dom))
        if cod and not isinstance(cod, Qudit):
            raise TypeError(messages.type_err(Qudit, cod))
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
