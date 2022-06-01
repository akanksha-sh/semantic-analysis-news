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
import utils
import getVisualData

"""Load Data"""
def dataloader():
    paths  = ['./out/', './data', './data/json']
    for path in paths:
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path)

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

    word_embedding_model = api.load('glove-wiki-gigaword-300')
    print("word embedding model loaded")

    sentiment_predictor = Predictor.from_path("/vol/bitbucket/as16418/tempFolder/models/roberta-sentiment.tar.gz", cuda_device=torch.cuda.current_device())
    print("sentiment model loaded")

    cf = Predictor.from_path("/vol/bitbucket/as16418/tempFolder/models/coref-spanbert-large.tar.gz", cuda_device=torch.cuda.current_device())
    print("coref model loaded")
    
    ner_predictor = Predictor.from_path("/vol/bitbucket/as16418/tempFolder/models/fine-grained-ner.tar.gz", cuda_device=torch.cuda.current_device())
    print("ner model loaded")
    
    return cf, sp, word_embedding_model, ner_predictor, sentiment_predictor


def run(cf, sp, word_embedding_model, pickled_ner, sentiment_predictor, input_group): 

    print("Unpickling ...")
    ner_predictor = dill.loads(pickled_ner)

    file_suffix = '{0}-{1}'.format(input_group.category[0], input_group.date[0].year)
    os.makedirs('./out/{0}/'.format(file_suffix))
    os.makedirs('./data/json/{0}/'.format(file_suffix))

    """Load Data"""
    articles = input_group['article']
    titles = input_group['title']
    no_sentences = 5
    intros = dataProcessing.get_intros(titles, articles, sp, no_sentences)
    print("No of sentences", no_sentences)


    """ Coref Resolution """
    coref_intros = [cf.coref_resolved(i) for i in intros]


    """ Preprocessing """
    coref_intros_filt = [dataProcessing.clean_text(i) for i in coref_intros]
    remove_entities_intros = [dataProcessing.remove_entities(ner_predictor, d) for d in coref_intros_filt]


    """ Tokenise and lemmatise """
    added_stopwords = {'airline', 'flight', 'from', 'subject', 're', 'say', "said", "would", "also",  "says", "read"}
    stopwords_sp = sp.Defaults.stop_words
    updated_stopwords = stopwords_sp | added_stopwords
    filtered_tokens = [dataProcessing.get_filtered_tokens(d, sp, updated_stopwords) for d in remove_entities_intros]


    """ Document embeddings """
    vectorized_intros = np.array(clustering.vectorize(filtered_tokens, model=word_embedding_model))
    # vectorized_intros = np.array(clustering.vectorize_tfidf(filtered_tokens, model=word_embedding_model))

    print("vectorized intros")


    """ Get clusters """
    normalised_data = normalize(vectorized_intros, norm="l2")
    tranformed_data = clustering.transform_data(normalised_data, 2)
    k_labels, centroids, opt_cluster_no, sil_score = clustering.optimal_cluster_number_kmeans(tranformed_data)
    clustering.visualise_clusters(tranformed_data, k_labels, centroids, file_suffix)
    print("Optimal cluster number", opt_cluster_no)

    # return opt_cluster_no, sil_score

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
    article_df.to_json('./data/json/{0}/article_info.json'.format(file_suffix), orient = 'index', indent=2)
    utils.save_artcile_info(input_group.category[0], input_group.date[0].year, file_suffix, article_df)

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
    
        aug_file_suffix = '{0}/Cluster-{1}'.format(file_suffix, cid)

        # """Relation Extraction"""
        # df_triples_m2 = relationExtraction.get_data_triples(topics_to_docs, coref_intros, stopwords_sp, relationExtraction.extract_relations, sp)
        # df_triples_m2.to_csv('./out/triples_m2-{0}.csv'.format(file_suffix))
        # relationExtraction.draw_kg(df_triples_m2, 'rel-m2', file_suffix, show_rels=True)

        relationExtraction.get_data_triples(topic_doc_mapping, coref_intros, updated_stopwords, aug_file_suffix, relationExtraction.find_triplet, sp, ner_predictor, sentiment_predictor)

    joined_topics_data = list(itertools.chain(*topics_data))
    topics_df = pd.DataFrame(joined_topics_data, columns = ['ClusterId', 'TopicId','Topic Name', 'Keywords', 'Articles', 'Sentiment'])
    topics_df.set_index('TopicId', inplace=True)
    topics_grouped_df = topics_df.groupby('ClusterId').apply(lambda x: x.loc[:, x.columns != "ClusterId"].to_dict(orient='index')).to_json(orient='index')
    json.dump(json.loads(topics_grouped_df), open("./data/json/{0}/topics.json".format(file_suffix), "w"), indent=2)
    utils.save_topics(input_group.category[0], input_group.date[0].year, file_suffix, topics_df)

if __name__ == '__main__':
    torch.cuda.empty_cache()
    cf, sp, wem, ner, sen = load_models()
    pickled_ner = dill.dumps(ner, byref=True)
    print(len(pickled_ner))
    data_groups = dataloader()
    # res = {}
    for dg in data_groups:
        start = time.time()
        run(cf, sp, wem, pickled_ner, sen, dg)
        # o, s = run(cf, sp, wem, pickled_ner, sen, dg)
        # res[dg.category[0]+str(dg.date[0].year)] = (o,s)
        print("Time to run one group", time.time() - start)
    # print(res)
    getVisualData.getCombinedData()