import re
from spacy.matcher import Matcher
import itertools
import pandas as pd
from itertools import islice
from dataProcessing import clean_text, get_entities
import json
from topicModelling import get_sentiment


""" Util functions"""


def is_sublist(source, target):
    slen = len(source)
    return any(
        all(
            item1 == item2
            for (item1, item2) in zip(source, islice(target, i, i + slen))
        )
        for i in range(len(target) - slen + 1)
    )


def long_substr_by_word(data):
    subseq = []
    data_seqs = [s.split(" ") for s in data]
    if len(data_seqs) > 1 and len(data_seqs[0]) > 0:
        for i in range(len(data_seqs[0])):
            for j in range(len(data_seqs[0]) - i + 1):
                if j > len(subseq) and all(
                    is_sublist(data_seqs[0][i : i + j], x) for x in data_seqs
                ):
                    subseq = data_seqs[0][i : i + j]
    return " ".join(subseq)


def remove_stopwords(text, stopwords):
    str_stopwords = re.compile(r"\b(" + r"|".join(stopwords) + r")\b\s*")
    c = re.sub(str_stopwords, " ", text)
    c = re.sub(r"\s+", " ", c)
    return c


def filter_triples(triples, sentiment_predictor):
    filtered_triples = []
    # Commonly extracted from news articles but result in unmeaningful relations
    unwanted_relations = ["told", "said", "saying", "tell", "says", "saying", "say"]

    for (s, r, o, tId, aId) in triples:
        if (
            not all((s, r, o))
            or s == o
            or any(r.strip().find(i) != -1 for i in unwanted_relations)
        ):
            continue
        # Remove redunancy in triples
        lcs = long_substr_by_word([r, o])
        lcs2 = long_substr_by_word([s, o])
        o = o.replace(lcs, "").strip()
        o = o.replace(lcs2, "").strip()
        if o == "":
            continue
        # Get triple sentiment
        triple = " ".join([s, r, o])
        sent = get_sentiment(int(sentiment_predictor.predict(triple)["label"]))
        filtered_triples.append((s, r.strip(), o, tId, aId, sent))
    return filtered_triples


def get_data_triples(
    topics_to_docs, coref_intros, stopwords, file_suffix, function, *args
):
    topic_rels = []
    for tid, docs in topics_to_docs.items():
        rels = [
            function(
                remove_stopwords(clean_text(coref_intros[d]), stopwords), tid, d, *args
            )
            for d in docs
        ]
        topic_rels.append(list(itertools.chain(*rels)))

    concatenated_rels = list(itertools.chain(*topic_rels))

    if len(concatenated_rels) > 0:
        df_triples = pd.DataFrame(
            concatenated_rels,
            columns=[
                "source",
                "relation",
                "target",
                "topicId",
                "articleId",
                "sentiment",
            ],
        )
        df_triples.set_index("topicId", inplace=True)
        df_triples.to_csv("./out/{0}-triples.csv".format(file_suffix))

        grouped_rels = (
            df_triples.groupby(level=0)
            .apply(lambda x: json.loads(x.to_json(orient="records")))
            .to_dict()
        )
        json.dump(
            grouped_rels,
            open("./data/json/{0}-rels.json".format(file_suffix), "w"),
            indent=2,
        )


""" Relation Extraction: Final method """

verb_patterns = [
    [
        {"POS": "AUX", "OP": "?"},
        {"POS": "PART", "OP": "?"},
        {"POS": "VERB", "OP": "+"},
        {"POS": "ADP", "OP": "+"},
    ],
    [{"POS": "VERB", "OP": "?"}, {"POS": "ADV", "OP": "*"}, {"POS": "VERB", "OP": "+"}],
]


def find_root_of_sentence(doc):
    root_token = None
    for token in doc:
        if token.dep_ == "ROOT":
            root_token = token
    return root_token


def contains_root(verb_phrase, root):
    vp_start = verb_phrase.start
    vp_end = verb_phrase.end
    if root.i >= vp_start and root.i <= vp_end:
        return True
    else:
        return False


def get_verb_phrases(doc, sp_model):
    root = find_root_of_sentence(doc)
    matcher = Matcher(sp_model.vocab)
    matcher.add("verb-phrases", verb_patterns)
    matches = matcher(doc)
    # Get the substrings based on the position of Spans
    verb_phrases = [doc[start:end] for _, start, end in matches]
    new_vps = []
    for verb_phrase in verb_phrases:
        # Verb phrase only qualifies if it contains root
        if contains_root(verb_phrase, root):
            new_vps.append(verb_phrase)
    return new_vps


def get_subject_relation(verb_phrase, noun_phrases, ents):
    subject = None
    relation = None
    for n in noun_phrases:
        # Subject phrase must occur to the left of the root verb phrase
        if n.start < verb_phrase.start:
            valid_ents = [(i, n.text.find(i)) for i in ents if n.text.find(i) != -1]

            if len(valid_ents) == 0:
                continue
            valid_ents.sort(key=lambda tup: tup[1])
            # Gets the first occuring named entity in the sentence
            subject = valid_ents[0][0]
            # Predicate/Relation Augmentation
            relation = n.text.partition(subject)[2].strip() + " " + verb_phrase.text
            break
    return subject, relation


def find_object_phrase(verb_phrase, noun_phrases):
    for noun_phrase in noun_phrases:
        # Object phrase must occur to the right of the root verb phrase
        if noun_phrase.start > verb_phrase.start:
            return noun_phrase.text.strip()


def find_triplet(doc, tId, aId, sp_model, ner_predictor, sentiment_predictor):
    triples = []
    for s in sp_model(doc).sents:
        sent = sp_model(s.text.strip())
        verb_phrases = get_verb_phrases(sent, sp_model)
        if len(verb_phrases) == 0:
            continue
        # Get longest verb phrase
        verb_phrases.sort(key=len, reverse=True)
        verb_phrase = verb_phrases[0]

        ents = get_entities(
            ner_predictor.predict(sent.text.strip()),
            ignore_types=["DATE", "MONEY", "TIME", "CARDINAL", "PERCENT", "QUANTITY"],
        )
        noun_phrases = list(sent.noun_chunks)
        if len(ents) == 0 or len(noun_phrases) == 0:
            continue
        subject_phrase, relation = get_subject_relation(verb_phrase, noun_phrases, ents)
        object_phrase = find_object_phrase(verb_phrase, noun_phrases)
        triples.append((subject_phrase, relation, object_phrase, tId, aId))

    filtered_triples = filter_triples(triples, sentiment_predictor)

    return filtered_triples
