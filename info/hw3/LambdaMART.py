import time
from collections import defaultdict

import numpy as np

from sklearn.tree import DecisionTreeRegressor
from multiprocessing import Pool
import pickle
import datetime

import logging

# from numba import jit


# @jit
def dcg(scores):
    """
        Returns the DCG value of the list of scores.
        Parameters
        ----------
        scores : list
            Contains labels in a certain ranked order

        Returns
        -------
        DCG_val: int
            This is the value of the DCG on the given scores
    """
    return np.sum([(np.power(2, scores[i]) - 1) / np.log2(i + 2) for i in range(len(scores))])


# @jit
def dcg_k(scores, k):
    """
        Returns the DCG value of the list of scores and truncates to k values.
        Parameters
        ----------
        scores : list
            Contains labels in a certain ranked order
        k : int
            In the amount of values you want to only look at for computing DCG

        Returns
        -------
        DCG_val: int
            This is the value of the DCG on the given scores
    """
    return np.sum([
        (np.power(2, scores[i]) - 1) / np.log2(i + 2)
        for i in range(len(scores[:k]))
    ])


# @jit
def ideal_dcg(scores):
    """
        Returns the Ideal DCG value of the list of scores.
        Parameters
        ----------
        scores : list
            Contains labels in a certain ranked order

        Returns
        -------
        Ideal_DCG_val: int
            This is the value of the Ideal DCG on the given scores
    """
    scores = [score for score in sorted(scores)[::-1]]
    return dcg(scores)


# @jit
def ideal_dcg_k(scores, k):
    """
        Returns the Ideal DCG value of the list of scores and truncates to k values.
        Parameters
        ----------
        scores : list
            Contains labels in a certain ranked order
        k : int
            In the amount of values you want to only look at for computing DCG

        Returns
        -------
        Ideal_DCG_val: int
            This is the value of the Ideal DCG on the given scores
    """
    scores = [score for score in sorted(scores)[::-1]]
    return dcg_k(scores, k)


# @jit
def single_dcg(scores, i, j):
    """
        Returns the DCG value at a single point.
        Parameters
        ----------
        scores : list
            Contains labels in a certain ranked order
        i : int
            This points to the ith value in scores
        j : int
            This sets the ith value in scores to be the jth rank

        Returns
        -------
        Single_DCG: int
            This is the value of the DCG at a single point
    """
    return (np.power(2, scores[i]) - 1) / np.log2(j + 2)

# @jit
def get_pairs(true_scores):
    """
        Returns pairs of indexes where the first value in the pair has a higher score than the second value in the pair.
        Parameters
        ----------
        true_scores : list of int
            Contain a list of scores

        Returns
        -------
        query_pair : list of pairs
            This contains a list of pairs of indexes in scores.
    """

    for i in range(len(true_scores)):
        for j in range(len(true_scores)):
            if true_scores[i] > true_scores[j]:
                yield (i, j)


# @jit
def compute_lambda(args):
    """
        Returns the lambda and w values for a given query.
        Parameters
        ----------
        args : zipped value of true_scores, predicted_scores, good_ij_pairs, idcg, query_key
            Contains a list of the true labels of documents, list of the predicted labels of documents,
            i and j pairs where true_score[i] > true_score[j], idcg values, and query keys.

        Returns
        -------
        lambdas : numpy array
            This contains the calculated lambda values
        w : numpy array
            This contains the computed w values
        query_key : int
            This is the query id these values refer to
    """

    true_scores, predicted_scores, idcg, query_key = args
    num_docs = len(true_scores)
    sorted_indexes = np.argsort(predicted_scores)[::-1]
    rev_indexes = np.argsort(sorted_indexes)
    true_scores = true_scores[sorted_indexes]
    predicted_scores = predicted_scores[sorted_indexes]

    lambdas = np.zeros(num_docs)
    w = np.zeros(num_docs)

    pairs_num = 0
    single_dcgs = {}
    for i, j in get_pairs(true_scores):
        pairs_num += 1
        if (i, i) not in single_dcgs:
            single_dcgs[(i, i)] = single_dcg(true_scores, i, i)
        single_dcgs[(i, j)] = single_dcg(true_scores, i, j)
        if (j, j) not in single_dcgs:
            single_dcgs[(j, j)] = single_dcg(true_scores, j, j)
        single_dcgs[(j, i)] = single_dcg(true_scores, j, i)

    for i, j in get_pairs(true_scores):
        z_ndcg = abs(single_dcgs[(i, j)] - single_dcgs[(i, i)] + single_dcgs[(j, i)] - single_dcgs[(j, j)]) / idcg
        rho = 1 / (1 + np.exp(predicted_scores[i] - predicted_scores[j]))
        rho_complement = 1.0 - rho
        lambda_val = z_ndcg * rho
        lambdas[i] += lambda_val
        lambdas[j] -= lambda_val

        w_val = rho * rho_complement * z_ndcg
        w[i] += w_val
        w[j] += w_val
    return lambdas[rev_indexes], w[rev_indexes], query_key


# @jit
def group_queries(training_data, qid_index):
    """
        Returns a dictionary that groups the documents by their query ids.
        Parameters
        ----------
        training_data : Numpy array of lists
            Contains a list of document information. Each document's format is [relevance score, query index, feature vector]
        qid_index : int
            This is the index where the qid is located in the training data

        Returns
        -------
        query_indexes : dictionary
            The keys were the different query ids and teh values were the indexes in the training data that are associated of those keys.
    """
    query_indexes = defaultdict(list)
    index = 0
    for record in training_data:
        query_indexes[record[qid_index]].append(index)
        index += 1
    return query_indexes


class LambdaMART:
    def __init__(self, training_data=None, number_of_trees=10, learning_rate=0.1):
        """
        This is the constructor for the LambdaMART object.
        Parameters
        ----------
        training_data : list of int
            Contain a list of numbers
        number_of_trees : int (default: 5)
            Number of trees LambdaMART goes through
        learning_rate : float (default: 0.1)
            Rate at which we update our prediction with each tree
        """

        self.training_data = training_data
        self.number_of_trees = number_of_trees
        self.learning_rate = learning_rate
        self.trees = []

    def fit(self, test_data, save_period=10):
        """
        Fits the model on the training data.
        Returns
        -------
            yields scores for test data if it's not None.
        """
        start_time = time.time()

        logging.info('Running fit job.')
        query_indexes = group_queries(self.training_data, 1)
        logging.info('Queries grouped.')
        query_keys = query_indexes.keys()
        true_scores = [self.training_data[query_indexes[query], 0] for query in query_keys]
        logging.info('True scores obtained.')
        predicted_scores = np.ones(len(self.training_data)) * np.mean(self.training_data[:, 0])
        logging.info('Prediction defaults created.')

        # ideal dcg calculation
        idcg = [ideal_dcg(scores) for scores in true_scores]
        logging.info("Ideal dcg's calculated")

        tree_times = []
        for k in range(self.number_of_trees):
            start_tree_time = time.time()
            logging.info(f"Training {k + 1} tree...")
            lambdas = np.zeros(len(predicted_scores))
            w = np.zeros(len(predicted_scores))
            pred_scores = [predicted_scores[query_indexes[query]] for query in query_keys]

            pool = Pool()
            for lambda_val, w_val, query_key in pool.map(compute_lambda,
                                                         zip(true_scores, pred_scores, idcg, query_keys),
                                                         chunksize=1):
                indexes = query_indexes[query_key]
                lambdas[indexes] = lambda_val
                w[indexes] = w_val
            pool.close()
            logging.info('Lambdas calculated.')

            tree = DecisionTreeRegressor(max_depth=5)
            tree.fit(self.training_data[:, 2:], lambdas)
            logging.info('Tree constructed.')

            nodes = tree.tree_.apply(self.training_data[:, 2:].astype(np.float32))
            for n in set(nodes):
                up = np.sum(lambdas[nodes == n])
                down = np.sum(w[nodes == n])
                if down == 0:
                    print(f"{up}/{down}, node {n}")
                tree.tree_.value[n] = np.array([up / down]).reshape(
                    tree.tree_.value[n].shape)
            logging.info('Weights updated.')

            self.trees.append(tree)
            prediction = tree.predict(self.training_data[:, 2:])
            predicted_scores += prediction * self.learning_rate
            tree_times.append(time.time() - start_tree_time)

            if k % save_period == 0:
                self.save(f"temp/temp_model_{k}")
                logging.info("saved")

            logging.info(
                f"Training {k + 1} tree done. Time spent: {datetime.timedelta(seconds=tree_times[-1])}. " +
                f"Estimated time: {datetime.timedelta(seconds=np.mean (tree_times) * (self.number_of_trees - k -1))}.")

            yield predicted_scores, self.predict(test_data)
        logging.info(f"Done. Time Elapsed:{datetime.timedelta(seconds=time.time() - start_time)}")
        pass

    def predict(self, data, num_of_trees=None):
        """
        Predicts the scores for the test dataset.
        Parameters
        ----------
        data : Numpy array of documents
            Numpy array of documents with each document's format is [query index, feature vector]

        Returns
        -------
        predicted_scores : Numpy array of scores
            This contains an array or the predicted scores for the documents.
        """
        data = np.array(data)
        query_indexes = group_queries(data, 0)
        predicted_scores = np.ones(len(data))
        for query in query_indexes:
            results = np.ones(len(query_indexes[query]))
            for i, tree in enumerate (self.trees):
                results += self.learning_rate * tree.predict(data[query_indexes[query], 1:])
                if (num_of_trees is not None) and i > num_of_trees:
                    break
            predicted_scores[query_indexes[query]] = results
        return predicted_scores

    def validate(self, data, k):
        """
        Predicts the scores for the test dataset and calculates the NDCG value.
        Parameters
        ----------
        data : Numpy array of documents
            Numpy array of documents with each document's format is [relevance score, query index, feature vector]
        k : int
            this is used to compute the NDCG@k

        Returns
        -------
        average_ndcg : float
            This is the average NDCG value of all the queries
        predicted_scores : Numpy array of scores
            This contains an array or the predicted scores for the documents.
        """
        data = np.array(data)
        query_indexes = group_queries(data, 1)
        average_ndcg = []
        predicted_scores = np.zeros(len(data))
        for query in query_indexes:
            results = np.zeros(len(query_indexes[query]))
            for tree in self.trees:
                results += self.learning_rate * tree.predict(data[query_indexes[query], 2:])
            predicted_sorted_indexes = np.argsort(results)[::-1]
            t_results = data[query_indexes[query], 0]
            t_results = t_results[predicted_sorted_indexes]
            predicted_scores[query_indexes[query]] = results
            dcg_val = dcg_k(t_results, k)
            idcg_val = ideal_dcg_k(t_results, k)
            ndcg_val = (dcg_val / idcg_val)
            average_ndcg.append(ndcg_val)
        average_ndcg = np.nanmean(average_ndcg)
        return average_ndcg, predicted_scores

    def save(self, fname):
        """
        Saves the model into a ".lmart" file with the name given as a parameter.
        Parameters
        ----------
        fname : string
            Filename of the file you want to save

        """
        pickle.dump(self, open('%s.lmart' % (fname), "wb"), protocol=2)

    def load(self, fname):
        """
        Loads the model from the ".lmart" file given as a parameter.
        Parameters
        ----------
        fname : string
            Filename of the file you want to load

        """
        model = pickle.load(open(fname, "rb"))
        self.training_data = model.training_data
        self.number_of_trees = model.number_of_trees
        self.learning_rate = model.learning_rate
        self.trees = model.trees
