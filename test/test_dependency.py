import pickle

import pytest
from spacy.tests.util import get_doc
from spacy.vocab import Vocab

from discopy.grammar.dependency import autonomous_tree, autonomous_tree_nocaps


@pytest.fixture
def sentence():
    """ Setup spaCy sentence with dependency tree. """
    words = u"This is a test".split(" ")
    vocab = Vocab(strings=words)
    pos = ["DET", "VERB", "DET", "NOUN"]
    heads = [1, 0, 1, -2]

    doc = get_doc(vocab=vocab, words=words, pos=pos, heads=heads)
    sentence = next(doc.sents)

    return sentence


def test_autonomous_tree(sentence):
    assert autonomous_tree(sentence) ==\
        pickle.load(open("test/src/autonomous_tree.pickle", "rb"))
