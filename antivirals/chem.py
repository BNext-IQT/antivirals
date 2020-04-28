import pickle
from typing import Sequence, Generator
from multiprocessing import cpu_count
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from pysmiles.read_smiles import _tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import numpy as np


class _WrapGenerator:
    """
    Creates iterators out of replayable generators. Needed for gensim.
    """

    def __init__(self, func):
        self.func = func
        self.generator = func()

    def __iter__(self):
        self.generator = self.func()
        return self

    def __next__(self):
        res = next(self.generator)
        if not res:
            raise StopIteration
        else:
            return res


class Toxicity:
    """
    Implments the toxicity model ontop of the latent vectors of the chemical language model.
    """

    def __init__(self, language_model):
        self.language = language_model
        self.tox = RandomForestClassifier(
            bootstrap=False, criterion='entropy', max_features=0.25,
            min_samples_leaf=6, min_samples_split=6, n_estimators=256)

    def fit(self, X, Y):
        # Preallocate memory for performance
        latent_vecs = np.empty(
            (len(X), self.language.document_model.vector_size))

        for i, sent in enumerate(self.language.make_generator(X)):
            latent_vecs[i] = self.language.document_model.infer_vector(sent)

        self.tox.fit(latent_vecs, Y)

    def save(self, path):
        with open(path, 'wb') as fd:
            pickle.dump(self, fd)

    @staticmethod
    def load(path):
        with open(path, 'rb') as fd:
            return pickle.load(fd)


class Language:
    """
    A chemical language model that creates semantic latent vectors from chemicals, 
    based on the mutual information between subtokens of a chemical discriptor and 
    a surrigate prediction set encoding chemistry semantics.
    """

    def __init__(self, path=None):
        if path:
            self.document_model = Doc2Vec.load(path)

    def _smiles_to_trivial_lang(self, smiles_seq: Sequence[str]) -> Generator[str, None, None]:
        for smiles in smiles_seq:
            res = []
            for cat, symbol in _tokenize(smiles):
                res.append(cat.name + '_OF_' + str(symbol))
            yield ' '.join(res)

    def _smiles_to_advanced_lang(self, smiles_seq: Generator
                                 [str, None, None],
                                 training: bool = False) -> Generator[str, None,
                                                                      None]:
        for i, sent in enumerate(smiles_seq):
            sent: Sequence[str] = self._analyzer(sent)
            res: Sequence[str] = []
            for token in sent:
                if token in self.vocab:
                    res.append(token.replace(' ', '_AND_'))
            if training:
                yield TaggedDocument(words=res, tags=[i])
            else:
                yield res

    def _make_iterator(
            self, smiles_seq: Sequence[str],
            training: bool = False) -> _WrapGenerator:
        return _WrapGenerator(
            lambda: self._smiles_to_advanced_lang(
                self._smiles_to_trivial_lang(smiles_seq),
                training))

    def make_generator(self, X):
        return self._smiles_to_advanced_lang(self._smiles_to_trivial_lang(X))

    def to_vecs(self, smiles_seq: Generator[str, None, None]) -> Generator[np.ndarray, None, None]:
        translation = self.make_generator(smiles_seq)
        for sentence in translation:
            yield self.document_model.infer_vector(sentence)

    def fit(self, X_unmapped, X, Y, max_vocab=18000,
            max_features_to_test=180000, window=8, dims=32, max_ngram=5):
        cv = CountVectorizer(
            max_df=0.95, min_df=2, lowercase=False, ngram_range=(1, max_ngram),
            max_features=max_features_to_test,
            token_pattern='[a-zA-Z0-9$&+,:;=?@_/~#\\[\\]|<>.^*()%!-]+')

        X_vec = cv.fit_transform(self._smiles_to_trivial_lang(X))

        local_vocab = set()
        for feat in Y.columns:
            res = zip(cv.get_feature_names(),
                      mutual_info_classif(
                          X_vec, Y[feat], discrete_features=True)
                      )
            local_vocab.update(res)
        self.vocab = {i[0] for i in sorted(
            local_vocab, key=lambda i: i[1], reverse=True)[:max_vocab]}

        self._analyzer = cv.build_analyzer()

        generator = self._make_iterator(X_unmapped, training=True)

        document_model = Doc2Vec(
            vector_size=dims, workers=cpu_count(), window=window)
        document_model.build_vocab(generator)
        document_model.train(
            generator, total_examples=len(X_unmapped), epochs=36)

        self.document_model = document_model

    def save(self, path):
        with open(path, 'wb') as fd:
            pickle.dump(self, fd)

    @staticmethod
    def load(path):
        with open(path, 'rb') as fd:
            return pickle.load(fd)
