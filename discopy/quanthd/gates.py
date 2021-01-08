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

    def dagger(self):
        return Gate(
            self.name, self.dom, self.array,
            _dagger=None if self._dagger is None else not self._dagger)


def _e_k(n, k):
    v = [0] * n
    v[k] = 1
    return v


def _braket_array(*string, type_):
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
        self.array = _braket_array(*string, type_=cod)

    def dagger(self):
        return Bra(*self.string)


class Bra(Box):
    def __init__(self, *string, dom):
        dom, cod = _box_type(dom), ScalarType ** 0
        name = "Bra({})".format(', '.join(map(str, bitstring)))
        super().__init__(name, dom, cod)
        self.string = string
        self.array = _braket_array(*string, type_=dom)

    def dagger(self):
        return Ket(*self.string)


class X(Gate):
    """ Generalized X gate. """
    def __init__(self, dom):
        dom = _box_type(dom, exp_size=1, min_dim=2)
        super().__init__(name=f'X({dom[0]})', dom=dom)

    @property
    def array(self):
        d = self.dom[0]
        return np.eye(d)[:, (np.arange(d)+1) % d]

    # def dagger(self):
    #    d = self.dom[0]
    #    return type(self)(self.dom)**(d - 1)


class Neg(Gate):
    """ Negation gate. """
    def __init__(self, dom):
        dom = _box_type(dom, exp_size=1, min_dim=2)
        super().__init__(name=f'Neg({dom[0]})', dom=dom)

    @property
    def array(self):
        d = self.dom[0]
        return np.eye(d)[:, (d - np.arange(d)) % d]

    def dagger(self):
        # return type(self)(self.dom)**2
        c = type(self)(self.dom)
        return c >> c


class Z(Gate):
    """ Generalized Z gate. """
    def __init__(self, dom):
        dom = _box_type(dom, exp_size=1, min_dim=2)
        super().__init__(name=f'Z({dom[0]})', dom=dom)

    @property
    def array(self):
        d = self.dom[0]
        diag = np.exp(np.arange(d)*2j*np.pi/d)
        return np.diag(diag)

    # def dagger(self):
    #    d = self.dom[0]
    #    return type(self)(self.dom)**(d - 1)


class H(Gate):
    """
    Discrete Fourier transform gate. Note that in a qubit system this corresponds
    to the one-qubit Hadamard gate.
    """
    def __init__(self, dom):
        dom = _box_type(dom, exp_size=1)
        super().__init__(name=f'H({dom[0]})', dom=dom)

    @property
    def array(self):
        d = self.dom[0]
        m = (np.arange(d)*2j*np.pi/d)[..., np.newaxis]
        m = m @ np.arange(d)[np.newaxis]
        m = np.exp(m)/np.sqrt(d)
        return m
