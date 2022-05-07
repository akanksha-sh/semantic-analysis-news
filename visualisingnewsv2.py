""" Imports """
from pprint import pprint
import gensim
import gensim.downloader as api
import numpy as np
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim_models
import spacy
from allennlp.predictors.predictor import Predictor
from gensim.models import CoherenceModel

import dataProcessing
import relationExtraction
import topicModelling

""" Load Models """
cf = Predictor.from_path("./coref-spanbert-large.tar.gz")
print("coref model loaded")

sp = spacy.load("en_core_web_sm")
print("spacy model loaded")

# word2vec = api.load('word2vec-google-news-300')
# print("word2cev model loaded")

ner_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/fine-grained-ner.2021-02-11.tar.gz")
print("ner model loaded")

"""## Load Data"""
group = dataProcessing.get_group_data('Business', '2021-05-31 00:00:00+0000', '3M')
titles = group['title']
articles = group['article']
intros = dataProcessing.get_intros(articles)
print("intros")

""" Coref Resolution """
coref_intros = [cf.coref_resolved(i) for i in intros]
print("coref intros")

# """ Preprocessing """
stopwords_sp = sp.Defaults.stop_words
# coref_intros_filt = [dataProcessing.clean_text(i) for i in coref_intros]
# print("coref intros")

# """ Tokenise and lemmatise """
# filtered_tokens = dataProcessing.get_filtered_tokens(coref_intros_filt, stopwords_sp)
# print(*filtered_tokens,sep='\n\n')

# """ Word embeddings """
# vector_size = len(word2vec['flight'])
# vectorized_intros = np.array(topicModelling.vectorize(filtered_tokens, vector_size, model=word2vec))
# print(vectorized_intros.shape)

# """ Get clusters """
# # Get opt cluster number
# opt_cluster_no = 3
# transformed_data = np.array(topicModelling.transform_data(vectorized_intros, 2))
# print(transformed_data.shape)
# k_labels, k_centroids = topicModelling.visualise_clusters(transformed_data, opt_cluster_no)
# group['k_clusters'] = k_labels

# n_most_rep_docs=8
# intro_groups = group.groupby(['k_clusters'])
# cluster_to_docs, filtered_tokens_clusters =  topicModelling.get_cluster_docs(n_most_rep_docs, filtered_tokens, opt_cluster_no, intro_groups, transformed_data, k_centroids)
# print(cluster_to_docs)

# """ LDA """
# data_words_bigrams = topicModelling.make_bigrams(filtered_tokens_clusters)
# lda_dictionary, corpus_tfidf = topicModelling.get_lda_params(data_words_bigrams)

# # can use multicore / min prob tuning
# lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus_tfidf, id2word=lda_dictionary, num_topics=6, random_state=100, update_every=1, 
#                                            chunksize=5, minimum_probability = 0.2, passes=15, alpha='auto', per_word_topics=True)

# # Compute Coherence Score
# coherence_model_lda = CoherenceModel(model=lda_model, texts=data_words_bigrams, dictionary=lda_dictionary, coherence='c_v')
# coherence_lda = coherence_model_lda.get_coherence()
# print('Coherence Score: ', coherence_lda)

# pyLDAvis.enable_notebook()
# vis = pyLDAvis.gensim_models.prepare(lda_model, corpus_tfidf, lda_dictionary, mds='mmds')
# pyLDAvis.display(vis)

# doc_topics_dist = lda_model[corpus_tfidf]
# topic_cluster_mapping = topicModelling.get_topic_cluster_mapping(doc_topics_dist, cluster_to_docs)
# topics_df, topics_to_docs = topicModelling.get_topic_doc_mapping(topic_cluster_mapping, lda_model, cluster_to_docs)
# pprint(topics_df)

topics_to_docs = {1: [36, 7, 22, 43, 14, 34, 37, 29],
                  3: [28, 46, 44, 20, 21, 31, 3, 4],
                  5: [9, 15, 18, 26, 45, 40, 0, 10]}

""" NER Cooccurence """

all_rels_df = relationExtraction.get_data_triples(topics_to_docs, coref_intros, stopwords_sp, relationExtraction.get_cooccurences, sp, ner_predictor)
all_rels_df.to_csv('./out/cooccurences')

relationExtraction.draw_kg(all_rels_df, './out/coocc', show_rels=False)

###########################################################################################################################

df_triples_m1 = relationExtraction.get_data_triples(topics_to_docs, coref_intros, stopwords_sp, relationExtraction.get_entity_triples, sp, ner_predictor)
df_triples_m1.to_csv('./out/df_triples_m1')
relationExtraction.draw_kg(df_triples_m1, './out/fig_m1', show_rels=True)

df_triples_m2 = relationExtraction.get_data_triples(topics_to_docs, coref_intros, stopwords_sp, relationExtraction.extract_relations, sp)
df_triples_m2.to_csv('./out/df_triples_m2')
relationExtraction.draw_kg(df_triples_m2, './out/fig_m2', show_rels=True)

df_triples_m3 = relationExtraction.get_data_triples(topics_to_docs, coref_intros, stopwords_sp, relationExtraction.find_triplet, sp)
df_triples_m3.to_csv('./out/df_triples_m3')
relationExtraction.draw_kg(df_triples_m3, './out/fig_m3', show_rels=True)
