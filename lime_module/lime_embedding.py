"""
Functions for explaining text classifiers.
"""
from functools import partial
import itertools
import json
import re

import numpy as np
import scipy as sp
import sklearn
from sklearn.utils import check_random_state

from lime import explanation
from lime import lime_base

from torch.utils.data import DataLoader
from lime_module.embedding_indices_domain_mapper import EmbeddingList, EmbeddingListDomainMapper


class EmbeddingLimeExplainer(object):
    """Explains text classifiers.
       Currently, we are using an exponential kernel on cosine distance, and
       restricting explanations to words that are present in documents."""

    total_samples = 0

    def __init__(self,
                 kernel_width=25,
                 kernel=None,
                 verbose=False,
                 class_names=None,
                 idx_to_word=None,
                 feature_selection='auto',
                 mask_string=None,
                 random_state=None):
        """Init function.

        Args:
            kernel_width: kernel width for the exponential kernel.
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            class_names: list of class names, ordered according to whatever the
                classifier is using. If not present, class names will be '0',
                '1', ...
            idx_to_word: vocabulary of the corpus indexed with embedding indexes
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """

        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.random_state = check_random_state(random_state)
        self.base = lime_base.LimeBase(kernel_fn, verbose,
                                       random_state=self.random_state)
        self.class_names = class_names
        self.idx_to_word = idx_to_word
        self.feature_selection = feature_selection
        self.mask_string = mask_string


    def explain_instance(self,
                         document_in_embedding_list,
                         classifier_fn,
                         labels=(1,),
                         top_labels=None,
                         num_features = 10,
                         num_samples = 1000,
                         distance_metric='cosine',
                         model_regressor=None,
                         ):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly hiding features from
        the instance (see __data_labels_distance_mapping). We then learn
        locally weighted linear models on this neighborhood data to explain
        each of the classes in an interpretable way (see lime_base.py).

        Args:
            document_in_hierarchical_embedding_indexes: list of arrays of embedding indexes.
            classifier_fn: classifier prediction probability function, which
                takes a list of d strings and outputs a (d, k) numpy array with
                prediction probabilities, where k is the number of classes.
                For ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_samples: number of size of the neighborhood to learn the linear model
            upper_attention_vector: vector of attention weights. should sum to one
            distance_metric: the distance metric to use for sample weighting,
                defaults to cosine similarity
            model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase. Must have model_regressor.coef_
            and 'sample_weight' as a parameter to model_regressor.fit()
            raw_string_split_function: the function that will split the sentence into tokens associated to the idx_to_word
            bow: is the word considered in order?
        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """
        domain_mapper = EmbeddingListDomainMapper( EmbeddingList(document_in_embedding_list))
        
        data, yss, distances = self.__data_labels_distances(document_in_embedding_list, 
                                                            classifier_fn, 
                                                            num_samples,
                                                            distance_metric=distance_metric)
        if self.class_names is None:
            self.class_names = [str(x) for x in range(yss[0].shape[0])]
        ret_exp = explanation.Explanation(domain_mapper=domain_mapper,
                                          class_names=self.class_names,
                                          random_state=self.random_state)
        ret_exp.predict_proba = yss[0]
        if top_labels:
            labels = np.argsort(yss[0])[-top_labels:]
            ret_exp.top_labels = list(labels)
            ret_exp.top_labels.reverse()
        for label in labels:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score, ret_exp.local_pred) = self.base.explain_instance_with_data(
                data, yss, distances, label, num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)
        return ret_exp

    def __data_labels_distances(self,
                                embedding_list ,                          
                                classifier_fn,
                                num_samples,
                                distance_metric='cosine'):
        """Generates a neighborhood around a prediction.

        Generates neighborhood data by randomly removing words from
        the instance, and predicting with the classifier. Uses cosine distance
        to compute distances between original and perturbed instances.
        Args:
            indexed_string: the associated indexed string to the embedding
            flattened_embeddings: flattened list of embedding indexes, corresponding to the indexed_string
            sentences_lengths: initial lengths of each sentence in the document
            classifier_fn: classifier prediction probability function, which
                takes a string and outputs prediction probabilities. For
                ScikitClassifier, this is classifier.predict_proba.
                pytorch model, this is the model object, that can handle a generator, with the cross-entropy loss (for classification)
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for sample weighting,
                defaults to cosine similarity.


        Returns:
            A tuple (data, labels, distances), where:
                data: dense num_samples * K binary matrix, where K is the
                    number of tokens in indexed_string. The first row is the
                    original instance, and thus a row of ones.
                labels: num_samples * L matrix, where L is the number of target
                    labels
                distances: cosine distance between the original instance and
                    each perturbed instance (computed in the binary 'data'
                    matrix), times 100.
        """

        def distance_fn(x):
            return sklearn.metrics.pairwise.pairwise_distances(x, x[0], metric=distance_metric).ravel() * 100



        doc_size = len(embedding_list)

        sample = self.random_state.randint(1, doc_size , num_samples - 1) # this is where we would change for sampling using attention distribution, also cannot remove all words, not sure model would like that
        data = np.ones((num_samples, doc_size))
        data[0] = np.ones(doc_size)
        features_range = range(doc_size)
        sample_documents = []
        for i, size in enumerate(sample, start=1):
            inactive_index = self.random_state.choice(features_range, size, replace=False)
            new_doc = [] 
            for position in features_range:
                if position not in inactive_index:
                    new_doc.append(embedding_list[position])
                else:
                    data[i, position] = 0
                    
            sample_documents.append(new_doc)

        
        
        dataloader = DataLoader(sample_documents, batch_size=24)
        output = []
        for batch in dataloader:
            predict_proba = classifier_fn(batch) 
            output.append(predict_proba.cpu().detach().numpy().tolist())
            

        distances = distance_fn(sp.sparse.csr_matrix(data))
        return data, output, distances