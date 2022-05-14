from math import sqrt
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import itertools


""" K-Means Clustering """
def clustering_k_means(X, n_clusters): 
  kmeans = KMeans(n_clusters=n_clusters, n_init=25, random_state=0)
  labels  = kmeans.fit_predict(X)
  sil_score = silhouette_score(X, labels)
  return (labels, kmeans.cluster_centers_, sil_score)
  
def optimal_cluster_number_kmeans(X):
  (n_d, n_v) = X.shape
  # alt1: n/root n, alt 2: n//3 
  n = int(n_d * n_v / sqrt(n_d))
  print(n_d, n_v, n)
  results = [clustering_k_means(X, i) for i in range(2, n)]
  (l,c,s) = max(results,key=lambda item:item[2])
  print(s)
  return l, c, len(c)

"""Transform the data"""
def transform_data(X, dim):
  pca = PCA(dim)
  transformed_data = pca.fit_transform(X)
  return transformed_data

def visualise_clusters(X, k_labels, centroids, file_suffix):
  plt.scatter(X[:, 0], X[:, 1], c=k_labels, alpha=0.5, s=100)
  plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=20, color='black') 
  plt.savefig('./out/clustering-{0}'.format(file_suffix))

""" Word embedding"""
def vectorize(list_of_docs, model):
    features = []
    vector_size = len(model['flight'])

    for tokens in list_of_docs:
        zero_vector = np.zeros(vector_size)
        vectors = []
        for token in tokens:
            try:
                vectors.append(model[token])
            except KeyError:
                continue
        if vectors:
            vectors = np.asarray(vectors)
            avg_vec = vectors.mean(axis=0)
            'TODO: use tfidf equivalent? ask maryam, check mean is coreect axis'

            features.append(avg_vec)
        else:
            features.append(zero_vector)
    return features

def n_most_representative_for_cluster(n, t_data, cluster_id, groups, centroids):
    group_indices = np.array(groups.get_group(cluster_id).index.tolist())

    """Can omit clusters with less then m docs"""
    if len(group_indices) < 2:
      return []
    most_representative_indices = np.argsort(
      np.linalg.norm(t_data[group_indices] - centroids[cluster_id], axis=1))[:n]

    return group_indices[most_representative_indices]


def get_n_param(cluster_member_counts):
  n_range= cluster_member_counts.max() - cluster_member_counts.min()
  print(n_range)
  n_mean= int(cluster_member_counts.mean())
  print(n_mean)
  'FIXME: This is wrong, maybe use std deviation '

  return min(n_range, n_mean)

def get_cluster_docs(n_most_rep_docs, filtered_tokens, opt_cluster_no, intro_groups, transformed_data, k_centroids):
  """Get filtered tokens per cluster"""
  cluster_to_docs = {} 
  filtered_tokens_clusters = []
  for cid in range(opt_cluster_no):
      res = n_most_representative_for_cluster(n_most_rep_docs, transformed_data, cid, intro_groups, k_centroids)
      if len(res) == 0:
        continue
      cluster_to_docs[cid] = res

      #unsure!!!!
      lemmatised_lists = [filtered_tokens[i] for i in res]
      filtered_tokens_clusters.append(list(itertools.chain(*lemmatised_lists)))

  return cluster_to_docs, filtered_tokens_clusters
