import numpy as np
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel, LdaModel, TfidfModel
import itertools
import pandas as pd


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

  'TODO: Should I be using tfidf at all'
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

  """Sentiment analysis"""

def get_sentiment(a):
  # label: 1 -> positive, 0->  negatice
  if a >= 0.4 and a <= 0.6:
    return "Neutral"
  return "Positive" if a > 0.6 else "Negative"

def get_doc_sentiments(docs, sent_predictor, data):
  doc_sentiments = {}
  for d in docs:
      s = int(sent_predictor.predict(data['title'][d])['label'])
      doc_sentiments[d] = s
  return doc_sentiments

def get_topic_doc_mapping(topic_cluster_mapping, doc_sentiments, lda_model, cluster_to_docs, sp, embedding_model):
  keywords = []
  topics = []
  topic_sentiments = []
  topics_to_docs = {}

  for t, cs in topic_cluster_mapping.items():
      topic_keywords = [w for w, _ in lda_model.show_topic(t)]
      keywords.append(topic_keywords)
      topics.append(get_topic_name(topic_keywords, sp, embedding_model))
      docs = [cluster_to_docs.get(cid) for cid in cs]
      docs = list(itertools.chain(*docs))
      topics_to_docs[t] = docs
      avg_sent = np.mean(np.array([doc_sentiments[d] for d in docs]))
      topic_sentiments.append(get_sentiment(avg_sent))


  topics_df = pd.DataFrame(list(topic_cluster_mapping.items()), columns = ['TopicId','Clusters'])
  topics_df['Topics'] = topics
  topics_df['Keywords'] = keywords
  topics_df['Sentiment'] = topic_sentiments
  return topics_df, topics_to_docs