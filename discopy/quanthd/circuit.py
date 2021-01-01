# -*- coding: utf-8 -*-

from discopy import messages, monoidal, rigid, tensor
from discopy.cat import AxiomError
from discopy.rigid import Ob, Ty, Diagram
from discopy.tensor import np, Dim, Tensor


class Qudit(Ty):
    @staticmethod
    def prepare(*dims):
        return Qudit(*map(lamda v: ('qudit', int(v), dims)))

    @property
    def l(self):
        return BitsAndQubits(*self.objects[::-1])

    r = l
