import pickle
from sklearn.cluster import KMeans
import numpy as np
from joblib import Parallel, delayed
import pandas as pd


def kMeansRes(scaled_data, k, alpha_k=0.02):
    '''
    Parameters
    ----------
    scaled_data: matrix
        scaled data. rows are samples and columns are features for clustering
    k: int
        current k for applying KMeans
    alpha_k: float
        manually tuned factor that gives penalty to the number of clusters
    Returns
    -------
    scaled_inertia: float
        scaled inertia value for current k
    '''

    inertia_o = np.square((scaled_data - scaled_data.mean(axis=0))).sum()
    # fit k-means
    kmeans = KMeans(n_clusters=k, random_state=42).fit(scaled_data)
    scaled_inertia = kmeans.inertia_ / inertia_o + alpha_k * k
    return scaled_inertia


def chooseBestKforKMeansParallel(scaled_data, k_range):
    '''
    Parameters
    ----------
    scaled_data: matrix
        scaled data. rows are samples and columns are features for clustering
    k_range: list of integers
        k range for applying KMeans
    Returns
    -------
    best_k: int
        chosen value of k out of the given k range.
        chosen k is k with the minimum scaled inertia value.
    results: pandas DataFrame
        adjusted inertia value for each k in k_range
    '''

    ans = Parallel(n_jobs=-1, verbose=10)(delayed(kMeansRes)(scaled_data, k) for k in k_range)
    ans = list(zip(k_range, ans))
    results = pd.DataFrame(ans, columns=['k', 'Scaled Inertia']).set_index('k')
    best_k = results.idxmin()[0]
    return best_k, results


# The players index (embeddings) is the same as the players_features.csv

with open("../data/player_embeddings.pickle", "wb") as f:
    embeddings = pickle.load(f)

best_k, results = chooseBestKforKMeansParallel(embeddings, range(1, embeddings.shape[0]//2))

model = KMeans(n_clusters=best_k, random_state=42)
clusters = model.fit_predict(embeddings)

# this way we actually reduce our search space of users

# we cluster the user that are actually similar