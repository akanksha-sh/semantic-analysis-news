
from functools import partial
import ray
from allennlp.predictors.predictor import Predictor
from ray.util.multiprocessing import Pool
import itertools
from math import floor
from multiprocessing import cpu_count
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

def load_models():
    cf = Predictor.from_path("./models/coref-spanbert-large.tar.gz")
    print("coref model loaded")
    cf_id = ray.put(cf)

    # print(cf._dataset_reader.get_distributed_info())
    sp = spacy.load("en_core_web_sm")
    print("spacy model loaded")
    sp_id = ray.put(sp)

    word_embedding_model = api.load('glove-wiki-gigaword-50')
    print("word embedding model loaded")
    glove_id = ray.put(word_embedding_model)

    ner_predictor = Predictor.from_path("./models/fine-grained-ner.tar.gz")
    print("ner model loaded")
    ner_id = ray.put(ner_predictor)

    sentiment_predictor = Predictor.from_path("./models/roberta-sentiment.tar.gz")
    print("sentiment model loaded")
    sent_id = ray.put(sentiment_predictor)

    return cf_id, sp_id, glove_id, ner_id, sent_id

def dataloader():
    output_path = './out/'
    # if not os.path.exists(output_path)
    shutil.rmtree(output_path, ignore_errors=True)
    os.makedirs(output_path)

    data = pd.read_csv('airline.csv', parse_dates=[1])
    data.columns = ['url', 'date', 'title', 'author', 'category', 'article']
    data['date'] = data['date'].dt.normalize()

    counts = data['category'].value_counts()
    mean_count = floor(counts.mean())
    filtered_data= data[data['category'].groupby(data['category']).transform('size') > mean_count]
    c = filtered_data.groupby([filtered_data.date.dt.year, 'category'])
    data_groups = [c.get_group(x) for x in c.groups]
    for dg in data_groups:
        dg.reset_index(drop=True, inplace=True)
    return data_groups

@ray.remote
def run(cf, sp, word_embedding_model, ner_predictor, sentiment_predictor, input_group): 
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
    return opt_cluster_no

if __name__ == '__main__':
    ray.init()
    cf_id, sp_id, glove_id, ner_id, sent_id = load_models()
    data_groups = dataloader()
    print(len(data_groups))
    # Store the array in the shared memory object store once
    # so it is not copied multiple times.

    result_ids = [run.remote(cf_id, sp_id, glove_id, ner_id, sent_id, i) for i in data_groups[:2]]
    output = ray.get(result_ids)
    print("Output", output)
    # pool = Pool()
    # result_ids = pool.map(partial(get_ner.remote, ner_id), array)
    # output = ray.get(result_ids)
    # print(output)
