from functools import partial
import json
import ray
from allennlp.predictors.predictor import Predictor
from ray.util.multiprocessing import Pool
import itertools
from math import floor
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
import utils


def load_models():
    cf = Predictor.from_path("./models/coref-spanbert-large.tar.gz")
    print("coref model loaded")
    cf_id = ray.put(cf)

    sp = spacy.load("en_core_web_sm")
    print("spacy model loaded")
    sp_id = ray.put(sp)

    word_embedding_model = api.load("glove-wiki-gigaword-50")
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
    output_path = "./out/"
    shutil.rmtree(output_path, ignore_errors=True)
    os.makedirs(output_path)

    data = pd.read_csv("airline.csv", parse_dates=[1])
    data.columns = ["url", "date", "title", "author", "category", "article"]
    data["date"] = data["date"].dt.normalize()

    counts = data["category"].value_counts()
    mean_count = floor(counts.mean())
    filtered_data = data[
        data["category"].groupby(data["category"]).transform("size") > mean_count
    ]
    c = filtered_data.groupby([filtered_data.date.dt.year, "category"])
    data_groups = [c.get_group(x) for x in c.groups]
    for dg in data_groups:
        dg.reset_index(drop=True, inplace=True)
    return data_groups


@ray.remote
def run(cf, sp, word_embedding_model, ner_predictor, sentiment_predictor, input_group):
    file_suffix = "{0}-{1}".format(input_group.category[0], input_group.date[0].year)
    os.makedirs("./out/{0}/".format(file_suffix))
    os.makedirs("./data/json/{0}/".format(file_suffix))
    print(file_suffix)

    # Load Data
    articles = input_group["article"]
    titles = input_group["title"]
    no_sentences = 8
    intros = dataProcessing.get_intros(titles, articles, sp, no_sentences)

    # Coref Resolution
    coref_intros = [cf.coref_resolved(i) for i in intros]

    # Preprocessing
    coref_intros_filt = [dataProcessing.clean_text(i) for i in coref_intros]
    remove_entities_intros = [
        dataProcessing.remove_entities(ner_predictor, d) for d in coref_intros_filt
    ]

    # Tokenise and lemmatise
    added_stopwords = {
        "airline",
        "flight",
        "from",
        "subject",
        "re",
        "say",
        "said",
        "would",
        "also",
        "says",
    }
    stopwords_sp = sp.Defaults.stop_words
    updated_stopwords = stopwords_sp | added_stopwords
    filtered_tokens = [
        dataProcessing.get_filtered_tokens(d, sp, updated_stopwords)
        for d in remove_entities_intros
    ]

    # Document embeddings
    vectorized_intros = np.array(
        clustering.vectorize_tfidf(filtered_tokens, model=word_embedding_model)
    )

    # Get semantic clusters
    normalised_data = normalize(vectorized_intros, norm="l2")
    tranformed_data = clustering.transform_data(normalised_data, 1)
    (
        k_labels,
        centroids,
        opt_cluster_no,
        sil_score,
    ) = clustering.optimal_cluster_number_kmeans(tranformed_data)
    print("Optimal cluster number", opt_cluster_no, sil_score)

    input_group["k_clusters"] = k_labels
    intro_groups = input_group.groupby(["k_clusters"])

    cluster_member_counts = input_group["k_clusters"].value_counts()
    print("cluster member counts", cluster_member_counts)

    n_most_rep_docs, cluster_to_docs = clustering.get_cluster_docs(
        cluster_member_counts, opt_cluster_no, intro_groups, tranformed_data, centroids
    )
    print("Cluster -> docs", cluster_to_docs)

    # Article Sentiment
    docs = list(itertools.chain(*cluster_to_docs.values()))
    doc_sentiments = topicModelling.get_doc_sentiments(
        docs, sentiment_predictor, input_group
    )

    article_df = input_group[["url", "title"]].filter(
        items=doc_sentiments.keys(), axis=0
    )
    article_df["sentiment"] = pd.Series(doc_sentiments).map(
        topicModelling.get_sentiment
    )
    article_df.to_json(
        "./data/json/{0}/article_info.json".format(file_suffix),
        orient="index",
        indent=2,
    )
    utils.save_artcile_info(
        input_group.category[0], input_group.date[0].year, file_suffix, article_df
    )

    # Topic Modelling: LDA
    min_no_topics = 2
    topics_data = []
    for cid, docs in cluster_to_docs.items():
        cluster_tokens = [filtered_tokens[d] for d in docs]
        data_words = topicModelling.make_bigrams(cluster_tokens)
        max_no_topics = ceil(len(docs) / 2)
        print("Max topics", max_no_topics)
        if max_no_topics <= min_no_topics:
            continue

        doc_topics_dist, lda_model = topicModelling.topic_modelling(
            data_words, max_no_topics, min_no_topics
        )
        topic_doc_mapping = topicModelling.get_topic_doc_mapping(
            cid, doc_topics_dist, cluster_to_docs
        )
        topics_data.append(
            topicModelling.get_topic_data(
                cid,
                topic_doc_mapping,
                doc_sentiments,
                lda_model,
                sp,
                word_embedding_model,
                updated_stopwords,
            )
        )
        aug_file_suffix = "{0}/Cluster-{1}".format(file_suffix, cid)

        # Semantic Triple Extraction
        triple_stopwords = updated_stopwords - {
            "by",
            "for",
            "n't",
            "n’t",
            "to",
            "by",
            "not",
        }
        relationExtraction.get_data_triples(
            topic_doc_mapping,
            coref_intros,
            triple_stopwords,
            aug_file_suffix,
            relationExtraction.find_triplet,
            sp,
            ner_predictor,
            sentiment_predictor,
        )

    joined_topics_data = list(itertools.chain(*topics_data))
    topics_df = pd.DataFrame(
        joined_topics_data,
        columns=[
            "ClusterId",
            "TopicId",
            "Topic Name",
            "Keywords",
            "Articles",
            "Sentiment",
        ],
    )
    topics_df.set_index("TopicId", inplace=True)

    # Store json data for visualisation
    topics_grouped_df = (
        topics_df.groupby("ClusterId")
        .apply(lambda x: x.loc[:, x.columns != "ClusterId"].to_dict(orient="index"))
        .to_json(orient="index")
    )
    json.dump(
        json.loads(topics_grouped_df),
        open("./data/json/{0}/topics.json".format(file_suffix), "w"),
        indent=2,
    )
    utils.save_topics(
        input_group.category[0], input_group.date[0].year, file_suffix, topics_df
    )


if __name__ == "__main__":
    ray.init()
    cf_id, sp_id, glove_id, ner_id, sent_id = load_models()
    data_groups = dataloader()
    print(len(data_groups))
    # Store the array in the shared memory object store once
    result_ids = [
        run.remote(cf_id, sp_id, glove_id, ner_id, sent_id, i) for i in data_groups[:2]
    ]
    output = ray.get(result_ids)
    print("Output", output)

    # pool = Pool(processes=(cpu_count() - 1))
    # pool.map(partial(run, cf, sp, wem, pickled_ner, sen), data_groups)
    # pool.close()
