import pickle

import pytest
from spacy.vocab import Vocab
from spacy.tokens import Doc

from discopy.grammar.dependency import autonomous_tree, autonomous_tree_nocaps


@pytest.fixture
def sentence():
    """ Setup spaCy sentence with dependency tree. """
    words = u"This is a test".split(" ")
    vocab = Vocab(strings=words)
    pos = ["DET", "VERB", "DET", "NOUN"]
    deps = ["nsubj", "ROOT", "det", "attr"]
    heads = [1, 1, 3, 1]

    doc = Doc(vocab, words=words, pos=pos, heads=heads, deps=deps)
    sentence = next(doc.sents)

    return sentence


def test_autonomous_tree(sentence):
    assert autonomous_tree(sentence) ==\
        pickle.load(open("test/src/autonomous_tree.pickle", "rb"))
