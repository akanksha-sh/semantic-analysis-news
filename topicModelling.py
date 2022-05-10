from multiprocessing import freeze_support
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel, LdaModel, TfidfModel, LdaMulticore
import itertools
import pandas as pd
from math import sqrt

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

def visualise_clusters(X, k_labels, centroids):
  plt.scatter(X[:, 0], X[:, 1], c=k_labels, alpha=0.5, s=100)
  plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=20, color='black') 
  plt.savefig("./out/clustering")

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

def make_bigrams(token_clusters, min_count=3, threshold=4):
  bigram = gensim.models.Phrases(token_clusters, min_count, threshold) 
  bigram_model = gensim.models.phrases.Phraser(bigram)
  return [bigram_model[t] for t in token_clusters]

def get_coherences(data_words, corpus_tfidf, lda_dictionary, n_topics):
  lda_model = LdaModel(corpus=corpus_tfidf,
                                        id2word=lda_dictionary, num_topics=n_topics,
                                        random_state=100,
                                        chunksize=5, minimum_probability = 0.1,
                                        passes=15, alpha='asymmetric', per_word_topics=True)
  
  # Note umass for now otherwise it breaks !!!!!!!
  coherence_model= CoherenceModel(model=lda_model, texts=data_words, dictionary=lda_dictionary, coherence='u_mass')
  coherence_lda = coherence_model.get_coherence()

  return coherence_lda

def topic_modelling(data_words ,max_topics=10):
  lda_dictionary = corpora.Dictionary(data_words) #just using bigrams for now
  # Term Document Frequency
  lda_corpus = [lda_dictionary.doc2bow(text) for text in data_words]
   # Create the TF-IDF model
  lda_tfidf = TfidfModel(lda_corpus)
  corpus_tfidf = lda_tfidf[lda_corpus]
  
  n_topics_coherence = np.array([get_coherences(data_words, corpus_tfidf, lda_dictionary, n) for n in range(2, max_topics)])
  print(n_topics_coherence)
  # optimal_number_topics =  max(n_topics_coherence.items(), key= lambda x: x[1])[0]
  optimal_number_topics = np.argmax(n_topics_coherence) + 2
  print("Optimal number of topics", optimal_number_topics)

  lda_model = LdaModel(corpus=corpus_tfidf,
                        id2word=lda_dictionary, num_topics=4,
                          random_state=100, update_every=1,
                            chunksize=5, minimum_probability = 0.2,
                              passes=15, alpha='auto', per_word_topics=True)
  
  return lda_model[corpus_tfidf], lda_model


"""Topic name inference - tentative"""
def check_invalid(kw_doc, sp, allowed_pos=['NOUN']):
  doc = sp(kw_doc)
  return any([kw.pos_ not in allowed_pos or kw.text.find('air') != -1 for kw in doc])
  
def get_topic_name(keywords, sp, embedding_model): 
  vecs = []
  for kw in keywords:
    kw = kw.replace('_', ' ')
    if check_invalid(kw, sp):
      continue
    vecs.append([w for w in kw.split()])
  
  vecs = list(itertools.chain(*vecs))
  print(vecs)
  topic_names = embedding_model.most_similar_cosmul(positive=vecs, topn=8)
  filtered_topic_names = [tn[0] for tn in topic_names if not check_invalid(tn[0], sp)]

  return "TBD" if len(filtered_topic_names) == 0 else filtered_topic_names[0]

""" Topic - cluster - doc mapping """
def get_topic_cluster_mapping(doc_topics_dist, cluster_to_docs):
  cluster_topic_mapping = zip(cluster_to_docs.keys(), doc_topics_dist)

  # can add this to dataframe after removing rows with omitted clusters (which have less than min no of elements)
  topic_cluster_mapping = {}
  for cid, doc in cluster_topic_mapping:
      """ Just picking one dominant topic"""
      t = sorted(doc[0], key=lambda x: (x[1]), reverse=True)[0][0]
      topic_cluster_mapping.setdefault(t, []).append(cid)
  
  return topic_cluster_mapping

def get_topic_doc_mapping(topic_cluster_mapping, lda_model, cluster_to_docs, sp, embedding_model):
  keywords = []
  topics = []
  topics_to_docs = {}

  for t, cs in topic_cluster_mapping.items():
      topic_keywords = [w for w, _ in lda_model.show_topic(t)]
      keywords.append(topic_keywords)
      topics.append(get_topic_name(topic_keywords, sp, embedding_model))
      docs = [cluster_to_docs.get(cid) for cid in cs]
      docs = list(itertools.chain(*docs))
      topics_to_docs[t] = docs

  topics_df = pd.DataFrame(list(topic_cluster_mapping.items()), columns = ['TopicId','Clusters'])
  topics_df['Topics'] = topics
  topics_df['Keywords'] = keywords

  return topics_df, topics_to_docs