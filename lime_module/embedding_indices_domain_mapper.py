"""
Functions for explaining text classifiers.
"""

import numpy as np


from lime import explanation

class EmbeddingListDomainMapper(explanation.DomainMapper):
    """Maps feature ids to words or word-positions"""

    def __init__(self, embedding_list):
        """Initializer.

        Args:
            indexed_string: lime_text.IndexedString, original string
        """
        self.embedding_list = embedding_list

    def map_exp_ids(self, exp, positions=False):
        """Maps ids to words or word-position strings.

        Args:
            exp: list of tuples [(id, weight), (id,weight)]
            positions: if True, also return word positions

        Returns:
            list of tuples (word, weight), or (word_positions, weight) if
            examples: ('bad', 1) or ('bad_3-6-12', 1)
        """
        if positions:
            exp = [('%s_%s' % (
                self.embedding_list.word(x[0]),
                '-'.join(map(str, x[0]))), x[1])
                   for x in exp]
        else:
            exp = [(self.embedding_list.word(x[0]), x[1]) for x in exp]
        return exp

class EmbeddingList(object):
    """String with various indexes."""

    def __init__(self, embedding_list):
        """Initializer.

        Args:
            hierarchical_embedding_index_list: list of embedding index associated to a
            idx_to_word array
            idx_to_word: dictionnary of index that links embedding index to words (from the likes of fasttext)
        """
        self.embedding_list = embedding_list
        
        self.as_list = embedding_list
        self.as_np = np.array(self.embedding_list)
        self.positions = []
        self.inverse_vocab = []

        for i, embedding_index in enumerate(embedding_list):

            self.positions.append(i)
            self.inverse_vocab.append(i)

        self.positions = np.array(self.positions)

    def raw_string(self):
        """Returns the original raw string"""
        return " ".join(self.inverse_vocab)

    def num_words(self):
        """Returns the number of tokens in the vocabulary for this document."""
        return len(self.inverse_vocab)

    def word(self, id_):
        """Returns the word that corresponds to id_ (int)"""
        return self.inverse_vocab[id_]
