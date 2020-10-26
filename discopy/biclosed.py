# -*- coding: utf-8 -*-

"""
Implements the free biclosed monoidal category.
"""

from discopy import messages, monoidal
from discopy.cat import AxiomError


class Ty(monoidal.Ty):
    """
    Objects in a free biclosed monoidal category.
    Generated by the following grammar:

        ty ::= Ty(name) | ty @ ty | ty >> ty | ty << ty

    Examples
    --------
    >>> x, y = Ty('x'), Ty('y')
    >>> print(y << x >> y)
    ((y << x) >> y)
    >>> print((y << x >> y) @ x)
    ((y << x) >> y) @ x
    """
    def __init__(self, name=None, left=None, right=None):
        self.left, self.right = left, right
        super().__init__(*(() if name is None else (name, )))

    def __repr__(self):
        return self.name

    def __lshift__(self, other):
        return Over(self, other)

    def __rshift__(self, other):
        return Under(self, other)


class Over(Ty):
    """ Forward slash types. """
    def __init__(self, left, right):
        name = "Over({}, {})".format(repr(left), repr(right))
        super().__init__(name, left, right)

    def __str__(self):
        return "({} << {})".format(str(self.left), str(self.right))


class Under(Ty):
    """ Backward slash types. """
    def __init__(self, left, right):
        name = "Under({}, {})".format(repr(left), repr(right))
        super().__init__(name, left, right)

    def __str__(self):
        return "({} >> {})".format(str(self.left), str(self.right))


class Diagram(monoidal.Diagram):
    """ Diagrams in a biclosed monoidal category. """
    @staticmethod
    def id(dom):
        return Id(dom)

    @staticmethod
    def fa(left, right):
        """ Forward application. """
        return FA(left, right)

    @staticmethod
    def ba(left, right):
        """ Backward application. """
        return BA(left, right)


class Id(monoidal.Id, Diagram):
    """ Identity diagram in a biclosed monoidal category. """


class Box(monoidal.Box, Diagram):
    """ Boxes in a biclosed monoidal category. """


class FA(Box):
    """ Forward application. """
    def __init__(self, left, right):
        if not isinstance(left, Over) or left.right != right:
            raise AxiomError(messages.are_not_adjoints(left, right))
        dom, cod = left @ right, left.left
        super().__init__("FA({}, {})".format(left, right), dom, cod)
        self.left, self.right = left, right

    def __repr__(self):
        return "FA({}, {})".format(repr(self.left), repr(self.right))


class BA(Box):
    """ Backward application. """
    def __init__(self, left, right):
        if not isinstance(right, Under) or right.left != left:
            raise AxiomError(messages.are_not_adjoints(left, right))
        dom, cod = left @ right, right.right
        super().__init__("BA({}, {})".format(left, right), dom, cod)
        self.left, self.right = left, right

    def __repr__(self):
        return "BA({}, {})".format(repr(self.left), repr(self.right))


class Functor(monoidal.Functor):
    """
    Functors into biclosed monoidal categories.

    Examples
    --------
    >>> from discopy import rigid
    >>> x, y = Ty('x'), Ty('y')
    >>> F = Functor(
    ...     ob={x: x, y: y}, ar={},
    ...     ob_factory=rigid.Ty,
    ...     ar_factory=rigid.Diagram)
    >>> print(F(y << x >> y))
    y.r @ x @ y.l
    >>> assert F((y << x) >> y) == F(y << (x >> y))
    """
    def __init__(self, ob, ar, ob_factory=Ty, ar_factory=Diagram):
        super().__init__(ob, ar, ob_factory, ar_factory)

    def __call__(self, diagram):
        if isinstance(diagram, Over):
            return self(diagram.left) << self(diagram.right)
        if isinstance(diagram, Under):
            return self(diagram.left) >> self(diagram.right)
        if isinstance(diagram, FA):
            return self.ar_factory.fa(self(diagram.left), self(diagram.right))
        if isinstance(diagram, BA):
            return self.ar_factory.ba(self(diagram.left), self(diagram.right))
        return super().__call__(diagram)
