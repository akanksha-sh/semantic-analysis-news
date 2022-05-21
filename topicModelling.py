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

"""Optimal Topic Model"""
def get_coherences(data_words, data_corpus, lda_dictionary, n_topics):
  lda_model = LdaModel(corpus=data_corpus, id2word=lda_dictionary, num_topics=n_topics,
                                        random_state=100,
                                        chunksize=5, minimum_probability = 0.3,
                                        passes=20, alpha='asymmetric', per_word_topics=True)
  
  # Note umass for now otherwise it breaks !!!!!!!
  coherence_model= CoherenceModel(model=lda_model, texts=data_words, dictionary=lda_dictionary, coherence='c_v', processes=1)
  coherence_lda = coherence_model.get_coherence()

  return (lda_model, coherence_lda)

def topic_modelling(data_words, max_topics=10, min_topics=2):
  lda_dictionary = corpora.Dictionary(data_words) #just using bigrams for now
  # Term Document Frequency
  lda_corpus = [lda_dictionary.doc2bow(text) for text in data_words]
   # Create the TF-IDF model
  lda_tfidf = TfidfModel(lda_corpus)
  corpus_tfidf = lda_tfidf[lda_corpus]

  n_topics_coherence = [get_coherences(data_words, corpus_tfidf, lda_dictionary, n) for n in range(min_topics, max_topics)]
  models, coherences = list(zip(*n_topics_coherence))
  print(coherences)
  i = np.argmax(np.array([coherences]))
  lda_model = models[i]
  print("Optimal number of topics", i+min_topics)
  print("Lda coherence", coherences[i])
  
  return lda_model[corpus_tfidf], lda_model

"""Topic name inference"""
def check_invalid(kw_doc, sp, stopwords, model=None, model_check=False, allowed_pos=['NOUN']):
  kw = list(sp(kw_doc))[0]
  if model_check and kw.lemma_ not in model.vocab:
    return True 
    # return any([kw.pos_ not in allowed_pos or kw.text.find('air') != -1 for kw in doc])
  return kw.pos_ not in allowed_pos or kw.lemma_ in stopwords
  
def get_topic_name(keywords, sp, embedding_model, stopwords): 
  'TODO: Play around with different length of keywords '
  words = list(itertools.chain(*[kw.split('_') for kw in keywords]))
  # print("words", words)
  fw = [w for w in words if not check_invalid(w,sp, stopwords, model=embedding_model, model_check=True)][:5]
  # print("fw", fw)
  topic_names = embedding_model.most_similar_cosmul(positive=fw, topn=3)
  filtered_topic_names = [tn[0] for tn in topic_names if not check_invalid(tn[0], sp, stopwords)]
  return "TBD" if len(filtered_topic_names) == 0 else filtered_topic_names[0:3]

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

"""Get Topic Dataframe"""
def get_topic_doc_mapping(cluster_id, doc_topics_dist, cluster_to_docs):
  cluster_docs = cluster_to_docs[cluster_id]
  topic_doc_mapping = {}
  
  'TODO: Omit topics based on docs'
  # Get main topic in each document
  for i, row in enumerate(doc_topics_dist):
      t = sorted(row[0], key=lambda x: (x[1]), reverse=True)[0][0]
      topic_doc_mapping.setdefault(t, []).append(cluster_docs[i])
  
  return topic_doc_mapping

def get_topic_data(cid, topic_doc_mapping, doc_sentiments, lda_model, sp, embedding_model, stopwords):
  topics_data = []

  for t, docs in topic_doc_mapping.items():
      topic_keywords = [w for w, _ in lda_model.show_topic(t)]
      topic_name = get_topic_name(topic_keywords, sp, embedding_model, stopwords)
      avg_sent = np.mean(np.array([doc_sentiments[d] for d in docs]))
      topic_sentiment = get_sentiment(avg_sent)
      topics_data.append([cid, t, topic_name, topic_keywords, docs, topic_sentiment])

  return topics_data