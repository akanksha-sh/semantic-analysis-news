""" Imports """
# import pyLDAvis
# import pyLDAvis.gensim_models
from multiprocessing import Process, freeze_support
import os
import gensim.downloader as api

import numpy as np
import spacy
from allennlp.predictors.predictor import Predictor
from sklearn.preprocessing import normalize
import pprint
import dataProcessing
import relationExtraction
import topicModelling

""" Load Models """
cf = Predictor.from_path("./models/coref-spanbert-large.tar.gz")
print("coref model loaded")

sp = spacy.load("en_core_web_sm")
print("spacy model loaded")

word_embedding_model = api.load('glove-wiki-gigaword-50')
print("word embedding model loaded")

ner_predictor = Predictor.from_path("./models/fine-grained-ner.tar.gz")
print("ner model loaded")

# sentiment_predictor = Predictor.from_path("./models/roberta-sentiment.tar.gz")
# print("sentiment model loaded")

output_path = './out/'
if not os.path.exists(output_path):
    os.makedirs(output_path)

def run(): 
    """Load Data"""
    group = dataProcessing.get_group_data('Business', '2021-05-31 00:00:00+0000', '3M')
    titles = group['title']
    articles = group['article']
    intros = dataProcessing.get_intros(articles)
    print("intros")

    """ Coref Resolution """
    coref_intros = [cf.coref_resolved(i) for i in intros]
    print("coref intros")

    """ Preprocessing """
    stopwords_sp = sp.Defaults.stop_words
    coref_intros_filt = [dataProcessing.clean_text(i) for i in coref_intros]

    """ Tokenise and lemmatise """
    filtered_tokens = dataProcessing.get_filtered_tokens(coref_intros_filt, stopwords_sp)

    """ Word embeddings """
    vectorized_intros = np.array(topicModelling.vectorize(filtered_tokens, model=word_embedding_model))
    print("vectorized intros")

    """ Get clusters """
    normalised_data = normalize(vectorized_intros, norm="l2")
    tranformed_data = topicModelling.transform_data(normalised_data, 2)
    k_labels, centroids, opt_cluster_no= topicModelling.optimal_cluster_number_kmeans(tranformed_data)
    topicModelling.visualise_clusters(tranformed_data, k_labels, centroids)
    print("Optimal cluster number", opt_cluster_no)


    """Group by clusters"""
    group['k_clusters'] = k_labels
    intro_groups = group.groupby(['k_clusters'])

    n_most_rep_docs=10
    cluster_to_docs, filtered_tokens_clusters = topicModelling.get_cluster_docs(n_most_rep_docs, 
                                                                filtered_tokens, 
                                                                opt_cluster_no, 
                                                                intro_groups, 
                                                                tranformed_data, 
                                                                centroids)

    """ LDA """
    data_words = topicModelling.make_bigrams(filtered_tokens_clusters)

    doc_topics_dist, lda_model = topicModelling.topic_modelling(data_words, 4)
    print("topic modelling")
    topic_cluster_mapping = topicModelling.get_topic_cluster_mapping(doc_topics_dist, cluster_to_docs)
    topics_df, topics_to_docs = topicModelling.get_topic_doc_mapping(topic_cluster_mapping, lda_model, cluster_to_docs, sp, word_embedding_model)
    print(topics_to_docs)

    topics_df.to_csv('./out/topics_df.csv')

    """NER Cooccurence"""

    all_rels_df = relationExtraction.get_data_triples(topics_to_docs, coref_intros, stopwords_sp, relationExtraction.get_cooccurences, sp, ner_predictor)
    all_rels_df.to_csv('./out/cooccurences')
    relationExtraction.draw_kg(all_rels_df, './out/coocc', show_rels=False)

    """Relation Extraction"""

    df_triples_m1 = relationExtraction.get_data_triples(topics_to_docs, coref_intros, stopwords_sp, relationExtraction.get_entity_triples, sp, ner_predictor)
    df_triples_m1.to_csv('./out/df_triples_m1')
    relationExtraction.draw_kg(df_triples_m1, './out/fig_m1', show_rels=True)

    df_triples_m2 = relationExtraction.get_data_triples(topics_to_docs, coref_intros, stopwords_sp, relationExtraction.extract_relations, sp)
    df_triples_m2.to_csv('./out/df_triples_m2')
    relationExtraction.draw_kg(df_triples_m2, './out/fig_m2', show_rels=True)

    df_triples_m3 = relationExtraction.get_data_triples(topics_to_docs, coref_intros, stopwords_sp, relationExtraction.find_triplet, sp)
    df_triples_m3.to_csv('./out/df_triples_m3')
    relationExtraction.draw_kg(df_triples_m3, './out/fig_m3', show_rels=True)

if __name__ == '__main__':
    freeze_support()
    run()