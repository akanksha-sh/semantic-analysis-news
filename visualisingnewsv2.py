""" Imports """
# import pyLDAvis
# import pyLDAvis.gensim_models
from functools import partial
import itertools
from math import ceil, floor
from multiprocessing import Pool, cpu_count
import os
import shutil
import gensim.downloader as api
import time
import numpy as np
import spacy
from allennlp.predictors.predictor import Predictor
from sklearn.preprocessing import normalize
import torch
import dataProcessing
import relationExtraction
import topicModelling
import clustering
import pandas as pd
import dill
import json

"""Load Data"""
def dataloader():
    output_path = './out/'
    shutil.rmtree(output_path, ignore_errors=True)
    os.makedirs(output_path)

    """For json"""
    output_path_json = './out/json'
    shutil.rmtree(output_path, ignore_errors=True)
    os.makedirs(output_path)

    data = pd.read_csv('airline.csv', parse_dates=[1])
    data.columns = ['url', 'date', 'title', 'author', 'category', 'article']
    data['date'] = data['date'].dt.normalize()

    data['category'] = data['category'].fillna("Misc")
    data_groups = data.groupby([data.date.dt.year, 'category'])
    # print(data_groups['category'].value_counts())
    mean_count = floor(data_groups['category'].value_counts().mean())
    filtered_data_groups = [data_groups.get_group(x) for x in data_groups.groups if len(data_groups.get_group(x)) > mean_count]

    for dg in filtered_data_groups:
        dg.reset_index(drop=True, inplace=True)
    return filtered_data_groups

""" Load Models """
def load_models():
    sp = spacy.load("en_core_web_sm")
    print("spacy model loaded")

    'TODO: Change to gigaword 300'
    word_embedding_model = api.load('glove-wiki-gigaword-50')
    print("word embedding model loaded")
    # print(os.system('nvidia-smi'))

    sentiment_predictor = Predictor.from_path("/vol/bitbucket/as16418/tempFolder/models/roberta-sentiment.tar.gz", cuda_device=torch.cuda.current_device())
    print("sentiment model loaded")
    # print(os.system('nvidia-smi'))

    cf = Predictor.from_path("/vol/bitbucket/as16418/tempFolder/models/coref-spanbert-large.tar.gz", cuda_device=torch.cuda.current_device())
    print("coref model loaded")
    # print(os.system('nvidia-smi'))

    
    ner_predictor = Predictor.from_path("/vol/bitbucket/as16418/tempFolder/models/fine-grained-ner.tar.gz", cuda_device=torch.cuda.current_device())
    print("ner model loaded")
    # print(os.system('nvidia-smi'))
    
    return cf, sp, word_embedding_model, ner_predictor, sentiment_predictor


def run(cf, sp, word_embedding_model, pickled_ner, sentiment_predictor, input_group): 

    print("Unpickling ...")
    ner_predictor = dill.loads(pickled_ner)

    file_suffix = '{0}-{1}/'.format(input_group.category[0], input_group.date[0].year)
    os.makedirs('./out/{0}'.format(file_suffix))
    os.makedirs('./out/json/{0}'.format(file_suffix))

    """Load Data"""
    articles = input_group['article']
    intros = dataProcessing.get_intros(articles, sp)
    print("intros")


    """ Coref Resolution """
    coref_intros = [cf.coref_resolved(i) for i in intros]
    print("coref intros")


    """ Preprocessing """
    coref_intros_filt = [dataProcessing.clean_text(i) for i in coref_intros]
    ignore_types = ['TIME', 'QUANTITY']
    remove_entities_intros = [dataProcessing.remove_entities(ner_predictor, d, ignore_types) for d in coref_intros_filt]


    """ Tokenise and lemmatise """
    added_stopwords = {'airline', 'flight', 'from', 'subject', 're', 'say', "said", "would", "also"}
    stopwords_sp = sp.Defaults.stop_words
    updated_stopwords = stopwords_sp | added_stopwords
    filtered_tokens = [dataProcessing.get_filtered_tokens(d, sp, updated_stopwords) for d in remove_entities_intros]


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

    n_most_rep_docs, cluster_to_docs = clustering.get_cluster_docs(cluster_member_counts,
                                                            opt_cluster_no, 
                                                            intro_groups, 
                                                            tranformed_data, 
                                                            centroids)
    print(cluster_to_docs)

    """Sentiments"""
    docs = list(itertools.chain(*cluster_to_docs.values()))
    doc_sentiments = topicModelling.get_doc_sentiments(docs, sentiment_predictor, input_group)
    print(doc_sentiments)

    article_df = input_group[['url', 'title']].filter(items=doc_sentiments.keys(), axis=0)
    article_df["sentiment"] = pd.Series(doc_sentiments).map(topicModelling.get_sentiment)
    article_df.to_csv('./out/{0}article_info.csv'.format(file_suffix))
    article_df.to_json('./out/json/{0}article_info.json'.format(file_suffix), orient = 'index', indent=2)


    """ LDA """
    min_no_topics = 2
    topics_data = []
    for cid, docs in cluster_to_docs.items():
        cluster_tokens = [filtered_tokens[d] for d in docs]
        data_words = topicModelling.make_bigrams(cluster_tokens)
        max_no_topics = ceil(len(docs)/2)
        print(max_no_topics)
        if max_no_topics <= min_no_topics:
            continue

        doc_topics_dist, lda_model = topicModelling.topic_modelling(data_words, max_no_topics, min_no_topics)
        topic_doc_mapping = topicModelling.get_topic_doc_mapping(cid, doc_topics_dist, cluster_to_docs)
        topics_data.append(topicModelling.get_topic_data(cid, topic_doc_mapping, doc_sentiments, lda_model, sp, word_embedding_model, updated_stopwords))
    
        aug_file_suffix = '{0}Cluster-{1}'.format(file_suffix, cid)
        # os.makedirs(aug_file_suffix)

        # """NER Cooccurence"""
        # all_rels_df = relationExtraction.get_data_triples(topics_to_docs, coref_intros, stopwords_sp, relationExtraction.get_cooccurences, sp, ner_predictor)
        # all_rels_df.to_csv('./out/coocc-{0}.csv'.format(file_suffix))
        # relationExtraction.draw_kg(all_rels_df, 'coocc', file_suffix, show_rels=False)

        # """Relation Extraction"""

        # df_triples_m1 = relationExtraction.get_data_triples(topics_to_docs, coref_intros, stopwords_sp, relationExtraction.get_entity_triples, sp, ner_predictor)
        # df_triples_m1.to_csv('./out/triples_m1-{0}.csv'.format(file_suffix))
        # relationExtraction.draw_kg(df_triples_m1, 'rel_m1', file_suffix, show_rels=True)

        # df_triples_m2 = relationExtraction.get_data_triples(topics_to_docs, coref_intros, stopwords_sp, relationExtraction.extract_relations, sp)
        # df_triples_m2.to_csv('./out/triples_m2-{0}.csv'.format(file_suffix))
        # relationExtraction.draw_kg(df_triples_m2, 'rel-m2', file_suffix, show_rels=True)

        df_triples_m3 = relationExtraction.get_data_triples(topic_doc_mapping, coref_intros, stopwords_sp, aug_file_suffix, relationExtraction.find_triplet, sp, ner_predictor)


    joined_topics_data = list(itertools.chain(*topics_data))
    topics_df = pd.DataFrame(joined_topics_data, columns = ['ClusterId', 'TopicId','Topic Name', 'Keywords', 'Articles', 'Sentiment'])
    topics_df.set_index('TopicId', inplace=True)
    topics_df.to_csv('./out/{0}topics.csv'.format(file_suffix))

    topics_grouped_df = topics_df.groupby('ClusterId').apply(lambda x: x.loc[:, x.columns != "ClusterId"].to_dict(orient='index')).to_json(orient='index')
    json.dump(json.loads(topics_grouped_df), open("./out/json/{0}topics.json".format(file_suffix), "w"), indent=2)


if __name__ == '__main__':

    torch.cuda.empty_cache()
    cf, sp, wem, ner, sen = load_models()
    pickled_ner = dill.dumps(ner, byref=True)
    print(len(pickled_ner))
    data_groups = dataloader()
    for i in range(0,4):
        start = time.time()
        run(cf, sp, wem, pickled_ner, sen, data_groups[i])
        print("Time to run one group", time.time() - start)