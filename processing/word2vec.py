from gensim.test.utils import common_texts
from gensim.models import Word2Vec

import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from config import MODEL_LOCATION


def create_model(sentences=common_texts):
    model = Word2Vec(sentences=sentences, vector_size=2, min_count=1, window=5, workers=4, epochs=10)
    model.save(MODEL_LOCATION)


def get_vectors(words):
    model = Word2Vec.load(MODEL_LOCATION)
    vocab = list(model.wv.key_to_index)
    X = model.wv[vocab]
    print(X)
    # dbscan = TSNE(n_components=2)
    # X_dbscan = dbscan.fit_transform(X)
    df = pd.DataFrame(X, index=vocab, columns=['x', 'y'])
    print(df, X)
    fig = pyplot.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(df['x'], df['y'])
    for word, pos in df.iterrows():
        ax.annotate(word, pos)
    pyplot.show()
    # vectors = StandardScaler().fit_transform([model.wv[w][0] for w in words])
    return True
