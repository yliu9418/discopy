# -*- coding: utf-8 -*-

"""
Interface from spacy to discopy to create diagrams from a document.

Before using the interface, download a language model appropriate for
the task, see https://spacy.io/usage/models

In short, download the language model you would like::
    python -m spacy download en_core_web_sm

Then load it within python as follows::
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("The quick brown fox jumps over the lazy dog.")

or::
    python -m spacy download fr_core_news_sm
::
    nlp = spacy.load("fr_core_news_sm")
    doc = nlp("Le renard brun et agile saute par dessus le chien parasseux.")
"""

import functools

import spacy

from discopy import Diagram, Functor, Cap, Cup, Box, Id, Ty, Word

spacy_types = [Ty('ADJ'),
               Ty('ADP'),
               Ty('ADV'),
               Ty('AUX'),
               Ty('CONJ'),
               Ty('DET'),
               Ty('INTJ'),
               Ty('NOUN'),
               Ty('NUM'),
               Ty('PART'),
               Ty('PRON'),
               Ty('PROPN'),
               Ty('PUNCT'),
               Ty('SCONJ'),
               Ty('SYM'),
               Ty('VERB'),
               Ty('X'),
               Ty('SPACE'),
               Ty('CCONJ')]


# - Dependency trees following autonomization construction - #

def autonomous_tree(sentence):
    """
    Given a spaCy span object, recursively builds a tree in which the
    dependency links are encoded in 'leq' boxes, caps and cups. This
    construction is useful for keeping track of word order, and objects
    are limited to the spaCy part-of-speech tags, as opposed to
    (spaCy type, dependency) tuples as below.

    With snake removal, a simpler dependency tree can be obtained.

    example::
        diagram = autonomous_tree(sentence)
        diagram.draw()
        diagram.normal_form().draw()
    """

    add_caps_functor = Functor(ob={ty: ty for ty in spacy_types},
                               ar=_add_caps_box,
                               ob_factory=Ty,
                               ar_factory=Diagram)

    return add_caps_functor(autonomous_tree_nocaps(sentence.root))


def autonomous_tree_nocaps(token):
    """
    Helper function for autonomous_tree. Creates dependency tree
    without caps, which can then be mapped to one with caps using
    a functor, so that snake removal can be performed.
    """

    # recursion - base case
    if token.n_lefts + token.n_rights == 0:
        return Word(token.text, Ty(token.pos_))

    # recursion - self-reference
    # top layer - word, box and subtree
    left_cod = [Ty(l.pos_).r for l in token.lefts]
    left_cod.reverse()
    right_cod = [Ty(r.pos_).l for r in token.rights]
    right_cod.reverse()

    cod = left_cod + [Ty(token.pos_)] + right_cod
    cod = Ty().tensor(*cod)

    left_label = ", ".join([l.pos_ for l in token.lefts])
    right_label = ", ".join([r.pos_ for r in token.rights])

    label = f"{token.pos_}({left_label} * {right_label})"
    box = Box(label, Ty(token.pos_), cod)

    top_layer = Word(token.text, Ty(token.pos_)) >> box

    # subtree
    left_subtree = Id(Ty()).tensor(
        *[autonomous_tree_nocaps(l) for l in token.lefts])
    right_subtree = Id(Ty()).tensor(
        *[autonomous_tree_nocaps(r) for r in token.rights])

    top_layer = left_subtree @ top_layer @ right_subtree

    # bottom layer - cups and one Id
    left_cups = [Cup(l.l, l) for l in reversed(left_cod)]
    right_cups = [Cup(r, r.r) for r in right_cod]

    bottom_left = Id(Ty())
    left_wires = Id(Ty())
    right_wires = Id(Ty())
    for cup in left_cups:
        new = left_wires @ cup @ right_wires
        bottom_left = new >> bottom_left
        left_wires = left_wires @ Id(Ty(cup.dom[0]))
        right_wires = Id(Ty(cup.dom[1])) @ right_wires

    bottom_right = Id(Ty())
    left_wires = Id(Ty())
    right_wires = Id(Ty())
    for cup in right_cups:
        new = left_wires @ cup @ right_wires
        bottom_right = new >> bottom_right
        left_wires = left_wires @ Id(Ty(cup.dom[0]))
        right_wires = Id(Ty(cup.dom[1])) @ right_wires

    bottom_layer = bottom_left @ Id(Ty(token.pos_)) @ bottom_right

    return top_layer >> bottom_layer


def _add_caps_box(box):
    """
    Arrow mapping for the 'add caps' functor.
    """

    left = Id(Ty())
    right = Id(Ty())
    dom = Ty()
    rdom = Ty()
    cod = Ty()
    l = None
    m = Id(Ty())
    r = None
    for i, obj in enumerate(box.cod):
        if obj.z == 1:
            dom = Ty(obj.name) @ dom
            left = left @ Id(Ty(obj.name).r)
            if not l:
                l = Cap(Ty(obj.name).r, Ty(obj.name))
            else:
                l = (l >> Id(box.cod[:i]) @ Cap(Ty(obj.name).r, Ty(obj.name))
                     @ Id(Ty()).tensor(
                         *reversed([Id(Ty(obj.name))
                                    for obj in box.cod[:i].objects])))
        elif obj.z == 0:
            dom = dom @ Ty(obj.name)
            cod = cod @ Ty(obj.name)
            m = m @ Id(Ty(obj.name))
        elif obj.z == -1:
            rdom = Ty(obj.name) @ rdom
            right = right @ Id(Ty(obj.name).l)
            if not r:
                r = (Id(Ty()).tensor(
                     *reversed([Id(Ty(obj.name))
                                for obj in box.cod[i + 1:].objects]))
                     @ Id(box.cod[i + 1:]))\
                    >> (Id(Ty()).tensor(
                        *reversed([Id(Ty(obj.name))
                                   for obj in box.cod[i + 1:].objects]))
                        @ Cap(Ty(obj.name), Ty(obj.name).l)
                        @ Id(box.cod[i + 1:]))
            else:
                r = (Id(Ty()).tensor(
                     *reversed([Id(Ty(obj.name))
                                for obj in box.cod[i + 1:].objects]))
                     @ Cap(Ty(obj.name), Ty(obj.name).l) @ Id(box.cod[i + 1:])
                     >> r)

    dom = dom @ rdom
    full = m
    if l:
        full = l @ full
    if r:
        full = full @ r
    if box.dom != Ty():
        full = full >> (left @ Box(box.name, dom, cod) @ right)
    else:
        full = box >> full

    return full


# - Dependency Trees with objects as (spaCy type, dependency) tuples - #

def dependency_tree(token):
    """
    Given a spaCy token, recursively builds a dependency tree from its
    dependent children. This can be applied to the 'ROOT' token of a spaCy
    span object to create a dependency tree for an entire 'semantic chunk'.

    The objects of the diagram are tuples (part-of-speech, dependency) where
    dependency it the dependency of the token on its parent in the dependency
    tree ('ROOT' if the token is the head of the tree).

    example::
        sentences = [s for s in doc.sents]
        R = sentences[0].root
        dependency_tree(R).draw(figsize=(12,10))
    """

    # codomain is the token.dep_
    cod = Ty((token.pos_, token.dep_))

    # base case
    if token.n_lefts + token.n_rights == 0:
        return Box(token.text, Ty(), cod)

    else:
        # codomain starts as empty type
        dom = Ty()

        subdiagram = None
        for child in token.children:
            # build domain from dependent children
            dom = dom @ Ty((child.pos_, child.dep_))
            # build subdiagram recursively
            if subdiagram is not None:
                subdiagram = subdiagram @ dependency_tree(child)
            else:
                subdiagram = dependency_tree(child)

        # compose this token with the subdiagram
        return subdiagram >> Box(token.text, dom, cod)


def dependency_forest(doc):
    """
    Splits a spaCy document into sentences, then creates
    a dependency tree for each sentence.

    Returns
    -------
    forest : [monoidal.Diagram]
            python list of dependency trees
    """

    # split sentences
    sentences = [s for s in doc.sents]

    # get diagrams for each sentence
    forest = [dependency_tree(s.root) for s in sentences]

    return forest


# - Alternative Approach to dependency trees (probably less robust) - #

def assign_words(parsing, use_lemmas=False, generic_target=False):
    """
    Given a parsing of some text (a list of spaCy tokens with punctuation
    removed), uses the part-of-speech tags and dependency structure to assign
    codomains to each word.

    The p-o-s tags constitute a set of basic types,
    and the codomains of the words are formed from tensor products
    of these basic types and their adjoints.

    Parameters
    ----------

    use_lemmas : bool
        use lemmas for names of the Word boxes e.g. 'loves' becomes
        Word('love', type) this is useful for simplifying a functor from
        sentence to circuit (we have less words to specify the image of
        in the arrow mapping)

    generic_target : bool
        If True, a token with the 'ROOT' dependency tag will end up with
        a generic 's' type in its codomain, which may be useful in
        distinguishing it as a 'semantic output'.
    """

    # a list of words and a set of basic types for this parsing
    words = []
    types = set()
    # target type for eager_parse
    target_type = Ty('s')

    for token in parsing:

        # initial codomain
        if token.dep_ == "ROOT":
            ty = target_type = Ty('s') if generic_target else Ty(token.pos_)
        else:
            ty = Ty(token.pos_)
        # add to set of basic types (does nothing if type already in set)
        types.add(ty)

        # scan dependency tree, tensor codomain with appropriate adjoints
        if token.n_lefts:
            lefts = [Ty(token.pos_)
                     for token in token.lefts if token in parsing]
            if len(lefts):
                ty = functools.reduce(lambda x, y: x @ y, lefts).r @ ty

        if token.n_rights:
            rights = [Ty(token.pos_)
                      for token in token.rights if token in parsing]
            if len(rights):
                ty = ty @ functools.reduce(lambda x, y: x @ y, rights).l

        # add Word to list of Words in this parsing
        words.append(Word(token.lemma_ if use_lemmas else token.text, ty))

    return words, types, target_type


def sentence_to_diagram(parsing, **kwargs):
    """
    Takes a parsing of a sentence, assigns words to the tokens in the parsing,
    infers a set of basic types, and uses discopy's eager_parse
    to try and guess a diagram.
    """

    words, types, target = assign_words(parsing, **kwargs)

    try:
        diagram = eager_parse(*words, target=target)
    except NotImplementedError:
        # this avoids an error if one of the sentences cannot be diagram-ed in
        # 'document_to_diagrams'
        print(f"Could not infer diagram from spaCy's tags: {parsing}")
        return None

    return diagram, types


def document_to_diagrams(doc, drop_stop=False, **kwargs):
    """
    Splits a document into sentences, and tries to create diagrams for each
    sentence, and a creates a set of all the basic types that were inferred
    from the document.

    Drops punctuation, and stop words if drop_stop=True.
    """

    # split sentences and parse
    sentences = [s for s in doc.sents]
    sentence_parsings = [[
        token for token in s if (token.pos != spacy.symbols.PUNCT)
        and not (drop_stop and token.is_stop)] for s in sentences]

    # diagrams for each sentence and set of basic types for the whole document
    diagrams, types = map(list, zip(
        *[sentence_to_diagram(parsing, **kwargs)
          for parsing in sentence_parsings]))
    types = functools.reduce(lambda x, y: x.union(y), types)

    return diagrams, types
