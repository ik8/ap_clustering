from sklearn.cluster import AffinityPropagation, DBSCAN
import numpy as np
import pandas as pd

import distance
from dataset import FIT_SAMPLE
from processing.word2vec import create_model, get_vectors


def get_result_matrix(words, action=distance.levenshtein):
    return -1 * np.array([[action(w1, w2) for w1 in words] for w2 in words])


def main():
    # # model = AffinityPropagation(damping=0.5)
    # model = DBSCAN(eps=0.5, min_samples=20)
    # words = np.asarray(FIT_SAMPLE)
    # matrix = get_result_matrix(words)
    # model.fit(matrix)
    # print(matrix)
    # cluster_ids = np.unique(model.labels_)
    # for cluster_id in cluster_ids:
    #     # centroid = FIT_SAMPLE[model.cluster_centers_indices_[cluster_id]]
    #     # cluster = np.unique(words[np.nonzero(model.labels_ == cluster_id)])
    #     # cluster_items = ", ".join(cluster)
    #     print(f"{cluster_id}: 'a'")
    create_model(sentences=FIT_SAMPLE)
    vectors = get_vectors(words=FIT_SAMPLE)
    print(vectors)

if __name__ == '__main__':
    main()
