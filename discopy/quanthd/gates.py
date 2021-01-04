from discopy import messages
from discopy.cat import AxiomError
from discopy.tensor import np, Dim, Tensor
from discopy.quanthd.circuit import Qudit, Box, Sum, ScalarType, _box_type


def array_shape(dom, cod):
    dom = tuple(cod) + tuple(dom)
    t = map(lambda x: int(x.name), tuple(cod) + tuple(dom))
    return tuple(t)


class Gate(Box):
    def __init__(self, name, dom, array=None, data=None, _dagger=False):
        dom = _box_type(dom)
        if array is not None:
            self._array = np.array(array).reshape(array_shape(dom, dom))
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
    return np.reshape(tensor.array, array_shape(dom=ScalarType**0, cod=type_))


class Ket(Box):
    def __init__(self, *string, cod):
        dom, cod = ScalarType ** 0, _box_type(cod)
        name = "Ket({})".format(', '.join(map(str, string)))    # TODO Include dom
        super().__init__(name, dom, cod)
        self.string = string
        self.array = _ket_array(*string, type_=cod)

    def dagger(self):
        return Bra(*self.string)


class Bra(Box):
    def __init__(self, *string, dom):
        dom, cod = _box_type(dom), ScalarType ** 0
        name = "Bra({})".format(', '.join(map(str, bitstring)))
        super().__init__(name, dom, cod)
        self.string = string
        self.array = _ket_array(*string, type_=dom)

    def dagger(self):
        return Ket(*self.string)


class X(Gate):
    def __init__(self, dom):
        dom = _box_type(dom, exp_size=1)
        super().__init__(name=f'X({dom[0]})', dom=dom)

    # TODO dagger should be X**(d-1) (pow is sequential composition here)
    # TODO array


class Z(Gate):
    def __init__(self, dom):
        dom = _box_type(dom, exp_size=1)
        super().__init__(name=f'Z({dom[0]})', dom=dom)

    # TODO dagger
    # TODO array
