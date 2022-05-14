""" Imports """
# import pyLDAvis
# import pyLDAvis.gensim_models
from functools import partial
import itertools
from math import floor
from multiprocessing import Pool, Process, cpu_count, freeze_support
import os
import shutil
import gensim.downloader as api

import numpy as np
import spacy
from allennlp.predictors.predictor import Predictor
from sklearn.preprocessing import normalize

import dataProcessing
import relationExtraction
import topicModelling
import clustering
import pandas as pd
import dill

"""Load Data"""
def dataloader():
    output_path = './out/'
    shutil.rmtree(output_path, ignore_errors=True)
    os.makedirs(output_path)

    data = pd.read_csv('airline.csv', parse_dates=[1])
    data.columns = ['url', 'date', 'title', 'author', 'category', 'article']
    data['date'] = data['date'].dt.normalize()

    data['category'] = data['category'].fillna("Misc")
    data_groups = data.groupby([data.date.dt.year, 'category'])
    # print(data_groups['category'].value_counts())
    mean_count = floor(data_groups['category'].value_counts().mean())
    # print(mean_count)
    filtered_data_groups = [data_groups.get_group(x) for x in data_groups.groups if len(data_groups.get_group(x)) > mean_count]

    for dg in filtered_data_groups:
        dg.reset_index(drop=True, inplace=True)
    return filtered_data_groups


""" Load Models """
def load_models():
    cf = Predictor.from_path("./models/coref-spanbert-large.tar.gz")
    print("coref model loaded")
    # print(cf._dataset_reader.get_distributed_info())
    sp = spacy.load("en_core_web_sm")
    print("spacy model loaded")

    word_embedding_model = api.load('glove-wiki-gigaword-50')
    print("word embedding model loaded")
    
    ner_predictor = Predictor.from_path("./models/fine-grained-ner.tar.gz")
    print("ner model loaded")

    sentiment_predictor = Predictor.from_path("./models/roberta-sentiment.tar.gz")
    print("sentiment model loaded")
    
    return cf, sp, word_embedding_model, ner_predictor, sentiment_predictor

# def run(pickled_ner, input_group): 
def run(cf, sp, word_embedding_model, pickled_ner, sentiment_predictor, input_group): 
    print("Attempting to unpickle")
    ner_predictor = dill.loads(pickled_ner)

    file_suffix = '{0}-{1}'.format(input_group.category[0], input_group.date[0].year)

    """Load Data"""
    articles = input_group['article']
    intros = dataProcessing.get_intros(articles, sp)
    print("intros")

    """ Coref Resolution """
    coref_intros = [cf.coref_resolved(i) for i in intros]
    print("coref intros")

    """ Preprocessing """
    stopwords_sp = sp.Defaults.stop_words
    coref_intros_filt = [dataProcessing.clean_text(i) for i in coref_intros]

    """ Tokenise and lemmatise """
    filtered_tokens = [dataProcessing.get_filtered_tokens(d, sp, stopwords_sp) for d in coref_intros_filt]

    """ Word embeddings """
    vectorized_intros = np.array(clustering.vectorize(filtered_tokens, model=word_embedding_model))
    print("vectorized intros")

    """ Get clusters """
    normalised_data = normalize(vectorized_intros, norm="l2")
    tranformed_data = clustering.transform_data(normalised_data, 2)
    k_labels, centroids, opt_cluster_no= clustering.optimal_cluster_number_kmeans(tranformed_data)
    clustering.visualise_clusters(tranformed_data, k_labels, centroids, file_suffix)
    print("Optimal cluster number", opt_cluster_no)


    """Group by clusters"""
    input_group['k_clusters'] = k_labels
    intro_groups = input_group.groupby(['k_clusters'])

    cluster_member_counts = input_group['k_clusters'].value_counts()
    print(cluster_member_counts)
    n_most_rep_docs = clustering.get_n_param(cluster_member_counts)
    cluster_to_docs, filtered_tokens_clusters = clustering.get_cluster_docs(n_most_rep_docs, 
                                                                filtered_tokens, 
                                                                opt_cluster_no, 
                                                                intro_groups, 
                                                                tranformed_data, 
                                                                centroids)
    print(cluster_to_docs)

    docs = list(itertools.chain(*cluster_to_docs.values()))
    doc_sentiments = topicModelling.get_doc_sentiments(docs, sentiment_predictor, input_group)
    print(doc_sentiments)

    """ LDA """
    data_words = topicModelling.make_bigrams(filtered_tokens_clusters)

    doc_topics_dist, lda_model = topicModelling.topic_modelling(data_words, 4)
    print("topic modelling")
    topic_cluster_mapping = topicModelling.get_topic_cluster_mapping(doc_topics_dist, cluster_to_docs)
    topics_df, topics_to_docs = topicModelling.get_topic_doc_mapping(topic_cluster_mapping, doc_sentiments, lda_model, cluster_to_docs, sp, word_embedding_model)
    print(topics_to_docs)
    topics_df.to_csv('./out/topics_df-{0}.csv'.format(file_suffix))

    # """NER Cooccurence"""
    all_rels_df = relationExtraction.get_data_triples(topics_to_docs, coref_intros, stopwords_sp, relationExtraction.get_cooccurences, sp, ner_predictor)
    all_rels_df.to_csv('./out/coocc-{0}.csv'.format(file_suffix))
    relationExtraction.draw_kg(all_rels_df, 'coocc', file_suffix, show_rels=False)

    # """Relation Extraction"""

    # df_triples_m1 = relationExtraction.get_data_triples(topics_to_docs, coref_intros, stopwords_sp, relationExtraction.get_entity_triples, sp, ner_predictor)
    # df_triples_m1.to_csv('./out/triples_m1-{0}.csv'.format(file_suffix))
    # relationExtraction.draw_kg(df_triples_m1, 'rel_m1', file_suffix, show_rels=True)

    # df_triples_m2 = relationExtraction.get_data_triples(topics_to_docs, coref_intros, stopwords_sp, relationExtraction.extract_relations, sp)
    # df_triples_m2.to_csv('./out/triples_m2-{0}.csv'.format(file_suffix))
    # relationExtraction.draw_kg(df_triples_m2, 'rel-m2', file_suffix, show_rels=True)

    df_triples_m3 = relationExtraction.get_data_triples(topics_to_docs, coref_intros, stopwords_sp, relationExtraction.find_triplet, sp)
    df_triples_m3.to_csv('./out/triples_m3-{0}.csv'.format(file_suffix))
    relationExtraction.draw_kg(df_triples_m3, 'rel-m3', file_suffix, show_rels=True)

if __name__ == '__main__':
    cf, sp, wem, ner, sen = load_models()
    pickled_ner = dill.dumps(ner, byref=True)
    print(len(pickled_ner))
    data_groups = dataloader()
    pool = Pool(processes=(cpu_count()-1))
    pool.map(partial(run, cf, sp, wem, pickled_ner, sen), data_groups[:1])
    pool.close()


# def alt_data_loader():
#     'Only filtered by category'

#     output_path = './out/'
#     shutil.rmtree(output_path, ignore_errors=True)
#     os.makedirs(output_path)

#     data = pd.read_csv('airline.csv', parse_dates=[1])
#     data.columns = ['url', 'date', 'title', 'author', 'category', 'article']
#     data['date'] = data['date'].dt.normalize()

#     counts = data['category'].value_counts()
#     mean_count = floor(counts.mean())
#     filtered_data= data[data['category'].groupby(data['category']).transform('size') > mean_count]
#     c = filtered_data.groupby([filtered_data.date.dt.year, 'category'])
#     data_groups = [c.get_group(x) for x in c.groups]

#     for dg in data_groups:
#         dg.reset_index(drop=True, inplace=True)

# def get_data_groups(data):
#   counts = data['category'].value_counts()
#   mean_count = counts.mean()
#   filtered_data= data[data['category'].groupby(data['category']).transform('size')> mean_count]

#   data_groups = []
#   cat_group_by = filtered_data.groupby('category')
#   for x in cat_group_by.groups:
#     cat_group = cat_group_by.get_group(x)
#     cat_date_group_by = cat_group.groupby(pd.Grouper(key='date', freq='6M'))
#     for y in cat_date_group_by.groups:
#       cat_date_group = cat_date_group_by.get_group(y)
#       cat_date_group.reset_index(drop=True, inplace=True)
#       data_groups.append(cat_date_group)
#       print(x, y, len(cat_date_group))
#   return data_groups

