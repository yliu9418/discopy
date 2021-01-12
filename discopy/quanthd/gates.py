from discopy import messages
from discopy.cat import AxiomError
from discopy.tensor import np, Dim, Tensor
from discopy.quanthd.circuit import Qudit, Box, Sum, ScalarType, _box_type
from discopy.quanthd.circuit import Id, Circuit
from discopy.quantum.gates import format_number


def array_shape(dom, cod):
    assert isinstance(dom, Qudit) and isinstance(cod, Qudit)
    return tuple(cod) + tuple(dom)


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
        return Bra(*self.string, dom=self.cod)


class Bra(Box):
    def __init__(self, *string, dom):
        dom, cod = _box_type(dom), ScalarType ** 0
        name = "Bra({})".format(', '.join(map(str, string)))
        super().__init__(name, dom, cod)
        self.string = string
        self.array = _braket_array(*string, type_=dom)

    def dagger(self):
        return Ket(*self.string, cod=self.dom)


class X(Gate):
    """ Generalized X gate. """
    def __init__(self, dom):
        dom = _box_type(dom, exp_size=1, min_dim=2)
        super().__init__(name='X', dom=dom)

    @property
    def array(self):
        d = self.dom[0]
        return np.eye(d)[:, (np.arange(d)+1) % d]


class Neg(Gate):
    """ Negation gate. """
    def __init__(self, dom):
        dom = _box_type(dom, exp_size=1, min_dim=2)
        super().__init__(name='Neg', dom=dom)

    @property
    def array(self):
        d = self.dom[0]
        return np.eye(d)[:, (d - np.arange(d)) % d]

    def dagger(self):
        return type(self)(self.dom)


class Z(Gate):
    """ Generalized Z gate. """
    def __init__(self, dom):
        dom = _box_type(dom, exp_size=1, min_dim=2)
        super().__init__(name=f'Z', dom=dom)

    @property
    def array(self):
        d = self.dom[0]
        diag = np.exp(np.arange(d)*2j*np.pi/d)
        return np.diag(diag)


class H(Gate):
    """
    Discrete Fourier transform gate. Note that in a qubit system this corresponds
    to the one-qubit Hadamard gate.
    """
    def __init__(self, dom):
        dom = _box_type(dom, exp_size=1)
        super().__init__(name=f'H', dom=dom)

    @property
    def array(self):
        d = self.dom[0]
        m = (np.arange(d)*2j*np.pi/d)[..., np.newaxis]
        m = m @ np.arange(d)[np.newaxis]
        m = np.exp(m)/np.sqrt(d)
        return m


class Add(Gate):
    def __init__(self, dom):
        dom = _box_type(dom, exp_size=2)
        if dom[0] != dom[1]:
            raise ValueError('Qudits expected having same dimension')
        super().__init__(name=f'Add', dom=dom)

    @property
    def array(self):
        d = self.dom[0]
        p = np.mgrid[:d, :d].reshape((2, -1)).T
        p = np.sum(p, axis=1) % d
        p += np.repeat(np.arange(d)*d, d)
        return np.eye(len(p))[:, p]


def nadd(dom):
    """
    Create the NADD gate which corresponds to the Add gate followed by
    Neg applied to the least significant qudit.
    """
    dom = _box_type(dom, exp_size=2)
    if dom[0] <= 2:
        return Add(dom)
    return Add(dom) >> (Id(dom[1]) @ Neg(dom[0]))


def cups(t):
    """
    Nested CUPS.
    :param t: Leg type.
    """
    t = _box_type(t)
    return Box.cups(t, t)


def caps(t):
    return cups(t).dagger()


def trace(circuit):
    if not isinstance(circuit, Circuit):
        raise ValueError(f'Expected type Circuit, found {type(circuit)}')
    if circuit.dom != circuit.cod:
        raise ValueError('Expected circuit.dom == circuit.cod')

    while True:
        if len(circuit.dom) == 0:
            return circuit
        t1, t2 = circuit.dom[:1], circuit.dom[1:]
        circuit = (caps(t1) @ Id(t2)) >> (Id(t1) @ circuit) >> (cups(t1) @ Id(t2))


class Parametrized(Box):
    """
    Abstract class for parametrized boxes in a quantum circuit.
    """
    def __init__(self, name, dom, cod, data=None, **params):
        self._datatype = params.get('datatype', None)
        data = data\
            if getattr(data, "free_symbols", False) else self._datatype(data)
        self.drawing_name = '{}({})'.format(name, data)
        Box.__init__(self, name, dom, cod, data=data,
                     _dagger=params.get('_dagger', False))

    def subs(self, *args):
        return type(self)(super().subs(*args).data)

    @property
    def name(self):
        return '{}({})'.format(self._name, format_number(self.data))

    def __repr__(self):
        return self.name


class Scalar(Parametrized):
    """ Scalar, i.e. quantum gate with empty domain and codomain. """
    def __init__(self, data, datatype=complex, name=None):
        self.drawing_name = format_number(data)
        name = "scalar" if name is None else name
        dom, cod = ScalarType, ScalarType
        _dagger = None if data.conjugate() == data else False
        super().__init__(name, dom, cod,
                         datatype=datatype, data=data, _dagger=_dagger)

    @property
    def array(self):
        return [self.data]

    def grad(self, var):
        if var not in self.free_symbols:
            return Sum([], self.dom, self.cod)
        return Scalar(self.array[0].diff(var))

    def dagger(self):
        return self if self._dagger is None\
            else Scalar(self.array[0].conjugate())


SINGLE_QUDIT_GATE_CLASSES = [X, Z, H, Neg]
