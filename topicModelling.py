from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import numpy as np
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel, Phrases, phrases, ldamodel, TfidfModel
import itertools
import pandas as pd

"""## K-Means Clustering"""
def clustering_k_means(X, n_clusters): 
  normalised_data = normalize(X, norm="l2")
  kmeans = KMeans(n_clusters=n_clusters, n_init=25, random_state=0)
  kmeans.fit(X)

  sil_score = silhouette_score(X, kmeans.labels_)
  print("Cluster:" + str(n_clusters) + "sil_score" + str(sil_score))

  return kmeans.labels_, kmeans.cluster_centers_, sil_score


"""Transform the data"""
def transform_data(X, dim):
  pca = PCA(dim)
  transformed_data = pca.fit_transform(X)
  return transformed_data

def visualise_clusters(X, n_clusters):
  cluster_labels, centroids, _ = clustering_k_means(X, n_clusters)
  plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, alpha=0.5, s= 100)
  plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=20, color='red', edgecolor='black')
  plt.show()
  return cluster_labels, centroids


""" Word2Vec"""
def vectorize(list_of_docs, vector_size, model):
    features = []

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

"""Ngrams """

# values change them when more data.

def make_bigrams(token_clusters):
  bigram = gensim.models.Phrases(token_clusters, min_count=3, threshold=4) 
  bigram_model = gensim.models.phrases.Phraser(bigram)
  return [bigram_model[t] for t in token_clusters]

# def make_trigrams(token_clusters):
#     trigram = gensim.models.Phrases(bigram[token_clusters], threshold=4)  
#     trigram_model = gensim.models.phrases.Phraser(trigram)
#     return [trigram_model[bigram_model[t]] for t in token_clusters]

def get_lda_params(data_words):
  # Create Dictionary
  lda_dictionary = corpora.Dictionary(data_words) #just using bigrams for now

  # Term Document Frequency
  lda_corpus = [lda_dictionary.doc2bow(text) for text in data_words]

  # # Create the TF-IDF model
  lda_tfidf = TfidfModel(lda_corpus)
  corpus_tfidf = lda_tfidf[lda_corpus]

  return lda_dictionary, corpus_tfidf

def get_topic_cluster_mapping(doc_topics_dist, cluster_to_docs):
  cluster_topic_mapping = zip(cluster_to_docs.keys(), doc_topics_dist)

  # can add this to dataframe after removing rows with omitted clusters (which have less than min no of elements)
  topic_cluster_mapping = {}
  for cid, doc in cluster_topic_mapping:
      """ Just picking one dominant topic"""
      t = sorted(doc[0], key=lambda x: (x[1]), reverse=True)[0][0]
      topic_cluster_mapping.setdefault(t, []).append(cid)
  
  return topic_cluster_mapping

def get_topic_doc_mapping(topic_cluster_mapping, lda_model, cluster_to_docs):
  keywords = []
  topics_to_docs = {}
  for t, cs in topic_cluster_mapping.items():
      topic_keywords = ", ".join([w for w, p in lda_model.show_topic(t)])
      keywords.append(topic_keywords)
      docs = [cluster_to_docs.get(cid) for cid in cs]
      docs = list(itertools.chain(*docs))
      print("T", t, "docs:", docs)
      topics_to_docs[t] = docs

  topics_df = pd.DataFrame(list(topic_cluster_mapping.items()), columns = ['TopicId','Clusters'])
  topics_df['Keywords'] = keywords
  return topics_df, topics_to_docs