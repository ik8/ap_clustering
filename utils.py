import distance
import numpy as np


def get_similarity_matrix(words, action=distance.levenshtein):
    return -1 * np.array([[action(w1, w2) for w1 in words] for w2 in words])
