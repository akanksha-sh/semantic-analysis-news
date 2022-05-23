import re
import matplotlib.pyplot as plt
import networkx as nx
from spacy.matcher import Matcher 
import itertools
import spacy
import pandas as pd
from itertools import islice
from dataProcessing import clean_text, get_entities
import json 

""" Util functions"""

def is_sublist(source, target):
    slen = len(source)
    return any(all(item1 == item2 for (item1, item2) in zip(source, islice(target, i, i+slen))) for i in range(len(target) - slen + 1))

def long_substr_by_word(data):
    subseq = []
    data_seqs = [s.split(' ') for s in data]
    if len(data_seqs) > 1 and len(data_seqs[0]) > 0:
        for i in range(len(data_seqs[0])):
            for j in range(len(data_seqs[0])-i+1):
                if j > len(subseq) and all(is_sublist(data_seqs[0][i:i+j], x) for x in data_seqs):
                    subseq = data_seqs[0][i:i+j]
    return ' '.join(subseq)

def remove_stopwords(text, stopwords):
  str_stopwords = re.compile(r'\b(' + r'|'.join(stopwords) + r')\b\s*')
  c = re.sub(str_stopwords, ' ', text)
  c = re.sub(r"\s+", " ", c)
  return c

def filter_triples(triples):
  filtered_triples = []
  for (s, r, o, tId, aId) in triples:
    if not all((s, r, o)) or s == o or any(r.strip().find(i) != -1 for i in ["told", "said", "saying", "tell", "say"]):
      continue
    lcs = long_substr_by_word([s,o])
    if (len(lcs)> 3):
      # print("before s:", s)
      # print("before o:", o)
      # print('lcs%s' % lcs)

      o = o.replace(lcs, '').strip()
      # print("after:", o)

    filtered_triples.append((s, r, o, tId, aId))
  return filtered_triples

""" Get relations and draw visualisations """
def get_data_triples(topics_to_docs, coref_intros, stopwords, file_suffix, function, *args):
  topic_rels = []
  for tid, docs in topics_to_docs.items():
    rels = [function(remove_stopwords(clean_text(coref_intros[d]), stopwords), tid, d, *args) for d in docs]
    # rels = [function(clean_text(coref_intros[d]), topic_colouring[t], *args) for d in docs]
    topic_rels.append(list(itertools.chain(*rels)))

  concatenated_rels = list(itertools.chain(*topic_rels))
  if len(concatenated_rels) > 0:
    df_triples = pd.DataFrame(concatenated_rels, columns=['source', 'relation', 'target', 'topicId', 'articleId'])
    df_triples.set_index('topicId', inplace=True)
    df_triples.to_csv('./out/{0}-triples.csv'.format(file_suffix))

    grouped_rels = df_triples.groupby(level=0).apply(lambda x: json.loads(x.to_json(orient='records'))).to_dict()
    json.dump(grouped_rels, open("./out/json/{0}-rels.json".format(file_suffix), "w") , indent=2)

    # relationExtraction.draw_kg(df_triples_m3, 'rel-m3', file_suffix, show_rels=True)

""" Relation Extraction """

""" Method 3 """

verb_patterns = [[{'POS':'AUX', 'OP': '?'}, {"POS":"VERB"}, {"POS":"ADP"}], 
          [{'POS': 'VERB', 'OP': '?'},
           {'POS': 'ADV', 'OP': '*'},
           {'POS': 'VERB', 'OP': '+'}], 
           [{'POS': 'AUX', 'OP': '?'},{'POS': 'PART', 'OP': '?'},{'POS': 'VERB', 'OP': '+'}]
          ]

def find_root_of_sentence(doc):
    root_token = None
    for token in doc:
      if (token.dep_ == "ROOT"):
          root_token = token
    return root_token

def contains_root(verb_phrase, root):
    vp_start = verb_phrase.start
    vp_end = verb_phrase.end
    if (root.i >= vp_start and root.i <= vp_end):
        return True
    else:
        return False

def get_verb_phrases(doc, sp_model):
    root = find_root_of_sentence(doc)
    matcher = Matcher(sp_model.vocab) 
    matcher.add("verb-phrases", verb_patterns)
    matches = matcher(doc)
    verb_phrases = [doc[start:end] for _, start, end in matches] 
    new_vps = []
    for verb_phrase in verb_phrases:
        if (contains_root(verb_phrase, root)):
            new_vps.append(verb_phrase)
    return new_vps

def longer_verb_phrase(verb_phrases):
    longest_length = 0
    longest_verb_phrase = None
    for verb_phrase in verb_phrases:
        if len(verb_phrase) > longest_length:
            longest_verb_phrase = verb_phrase
    return longest_verb_phrase

def find_noun_phrase(verb_phrase, noun_phrases, side, ents):
    for noun_phrase in noun_phrases:
        # print(any([noun_phrase.text.find(i) != -1 for i in ents]))

        if (side == "left" and \
            noun_phrase.start < verb_phrase.start) and any([noun_phrase.text.find(i) != -1 for i in ents]):
            return noun_phrase.text.strip()
        elif (side == "right" and \
              noun_phrase.start > verb_phrase.start):
            return noun_phrase.text.strip()
  
def find_triplet(doc, tId, aId, sp_model, ner_predictor):
  triples = []
  for s in sp_model(doc).sents:
    sent = sp_model(s.text.strip())
    verb_phrases = get_verb_phrases(sent, sp_model)

    if len(verb_phrases) == 0:
      continue
    ignore_types = ['DATE', 'TIME', 'CARDINAL', 'MONEY', 'PERCENT', 'QUANTITY']
    ents = get_entities(ner_predictor.predict(sent.text.strip()), ignore_types= ignore_types)

    noun_phrases = sent.noun_chunks

    verb_phrase = None
    if (len(verb_phrases) > 1):
        verb_phrase = longer_verb_phrase(list(verb_phrases))
    else:
        verb_phrase = verb_phrases[0]

    left_noun_phrase = find_noun_phrase(verb_phrase, noun_phrases, "left", ents)
    right_noun_phrase = find_noun_phrase(verb_phrase, noun_phrases, "right", [])
  
    triples.append((left_noun_phrase, verb_phrase.text, right_noun_phrase, tId, aId))

  filtered_triples = filter_triples(triples)

  return filtered_triples

# def draw_kg(pairs, method, file_suffix, show_rels=True):
#   k_graph = nx.from_pandas_edgelist(pairs, 'subject', 'objects',create_using=nx.MultiDiGraph(), edge_attr='color')
#   node_deg = nx.degree(k_graph)
#   layout = nx.spring_layout(k_graph, k=1, iterations=60)
#   plt.figure(figsize=(35,30))
#   nx.draw_networkx(
#       k_graph,
#       node_size=[int(deg[1]) * 1000 for deg in node_deg],
#       linewidths=1.5,
#       pos=layout,
#       edge_color=nx.get_edge_attributes(k_graph,'color').values(),
#       edgecolors='black',
#       node_color='white',
#       )
#   if show_rels:
#     labels = dict(zip(list(zip(pairs.subject, pairs.objects)),pairs['relation'].tolist()))
#     nx.draw_networkx_edge_labels(k_graph, pos=layout, edge_labels=labels,font_color='black')
#   plt.savefig('./out/kg-{0}-{1}'.format(method, file_suffix))