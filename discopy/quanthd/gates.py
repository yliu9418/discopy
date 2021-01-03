from discopy import messages
from discopy.cat import AxiomError
from discopy.tensor import np, Dim, Tensor
from discopy.quanthd.circuit import Qudit, Box, Sum, ScalarType


class Gate(Box):
    def __init__(self, name, dom, array=None, data=None, _dagger=False):
        dom = Qudit(dom) is isinstance(dom, int) else dom
        if array is not None:
            # TODO Redo here
            self._array = np.array(array).reshape(dom.array_shape())
        super().__init__(name, dom, dom, data=data, _dagger=_dagger)

    @property
    def array(self):
        return self._array

    def __repr__(self):
        return "Gate({}, dom={}, array={})".format(
            repr(self.name), self.dom,
            np.array2string(self.array.flatten()))

    def dagger(self):   # TODO Note that gates should implement this...
        return Gate(
            self.name, self.dom, self.array,
            _dagger=None if self._dagger is None else not self._dagger)


def _e_k(n, k):
    v = [0] * n
    v[k] = 1
    return v


def _ket_array(*string, type_):
    if not isinstance(type_, Qudit):
        raise TypeError(messages.type_err(Qudit, type_))
    if len(string) != len(type_):
        raise ValueError('Mismatching string and type lengths')
    tensor = Tensor.id(Dim(1)).tensor(*(
        Tensor(Dim(1), Dim(n), _e_k(n, k)) for n, k in zip(type_, string)))
    return tensor.array


class Ket(Box):
    def __init__(self, *string, cod):
        dom, cod = ScalarType ** 0, cod
        name = "Ket({})".format(', '.join(map(str, string)))    # TODO Include dom
        super().__init__(name, dom, cod)
        self.string = string
        self.array = _ket_array(*string, type_=cod)

    def dagger(self):
        return Bra(*self.string)

    # TODO Add ket() and bra() on qubit type?


class Bra(Box):
    def __init__(self, *string, dom):
        name = "Bra({})".format(', '.join(map(str, bitstring)))
        dom, cod = dom, ScalarType ** 0
        super().__init__(name, dom, cod)
        self.string = string
        self.array = _ket_array(*string, type_=dom)

    def dagger(self):
        return Ket(*self.string)