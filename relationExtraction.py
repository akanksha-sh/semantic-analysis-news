import re
import matplotlib.pyplot as plt
import networkx as nx
from spacy.matcher import Matcher 
import itertools
import spacy
import pandas as pd
from itertools import islice
from dataProcessing import clean_text

def remove_stopwords(text, stopwords):
  str_stopwords = re.compile(r'\b(' + r'|'.join(stopwords) + r')\b\s*')
  c = re.sub(str_stopwords, ' ', text)
  c = re.sub(r"\s+", " ", c)
  return c

def get_entities(result):
  entities = set()
  ignore_types = ['DATE', 'TIME', 'CARDINAL', 'PERCENT', 'QUANTITY']
  for word, tag in zip(result["words"], result["tags"]):
      if tag == "O":
        continue
      ent_position, ent_type = tag.split("-")
      if ent_type in ignore_types:
        continue
      if ent_position == "U":
        entities.add(word)
      else:
        if ent_position == "B":
            e = word
        elif ent_position == "I":
            e += " " + word
        elif ent_position == "L":
            e += " " + word
            entities.add(e)

  return entities

def get_cooccurences(doc, col, sp_model, ner_pred):
  c_sents = []
  for s in sp_model(doc).sents:
    if len(s) < 3:
      continue
    ents = list(get_entities(ner_pred.predict(sentence=s.text.strip())))
    # get permuatations instead?
    if len(ents) != 2:
      continue
    c_sents.append((ents[0], ents[1], col))
  return c_sents


""" Relation Extraction """

""" Method 1 - entity triples"""

def get_relation(sent, sp_model):
    doc = sp_model(sent)
    matcher = Matcher(sp_model.vocab)

    #define the pattern 
    pattern = [{'DEP':'ROOT'}, 
            {'DEP':'prep','OP':"?"},
            {'DEP':'agent','OP':"?"},  
            {'POS':'ADJ','OP':"?"}] 

    matcher.add("matching_1", [pattern], on_match=None)

    matches = matcher(doc)
    k = len(matches) - 1
    span = doc[matches[k][1]:matches[k][2]] 
    return span.text

def get_entity_triples(doc, col, sp_model, ner_pred):
  triples = []
  for s in sp_model(doc).sents:
    if len(s) < 3:
      continue
    s = s.text.strip()
    ents = list(get_entities(ner_pred.predict(sentence=s)))
    if len(ents) != 2:
      continue
    rel = get_relation(s, sp_model)
    if not rel or rel.strip() in ["told", "said", "saying", "tell", "say"]:
      continue
    triples.append((ents[0], rel, ents[1], col))

  return triples

""" Method 2 """

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

def refine_ent(ent, sent, sp_model):
  unwanted_tokens = (
      'PRON',  # pronouns
      'PART',  # particle
      'DET',  # determiner
      'SCONJ',  # subordinating conjunction
      'PUNCT',  # punctuation
      'SYM',  # symbol
      'X',  # other
  )
  ent_type = ent.ent_type_  # get entity type
  if ent_type == '':
      ent_type = 'NOUN_CHUNK'
      ent = ' '.join(str(t.text) for t in sp_model(str(ent)) if t.pos_ not in unwanted_tokens and t.is_stop == False)
      
  elif ent_type in ('CARDINAL', 'ORDINAL') and str(ent).find(' ') == -1:
      refined = ''
      for i in range(len(sent) - ent.i):
          if ent.nbor(i).pos_ not in ('VERB', 'PUNCT'):
              refined += ' ' + str(ent.nbor(i))
          else:
              ent = refined.strip()
              break

  return ent, ent_type

def filter_triples(triples):
  filtered_triples = []
  for (s, r, o, c) in triples:
    if not all((s, r, o, c)) or s == o or r.strip() in ["told", "said", "saying", "tell", "say"]:
      continue
    lcs = long_substr_by_word([s,o])
    if (len(lcs)> 3):
      o = o.replace(lcs, '').strip()


    filtered_triples.append((s, r, o, c))
  return filtered_triples

def extract_relations(doc, col, sp_model):
  ent_pairs = []
  for s in sp_model(doc).sents:
    sent = s.text.strip()
    sent = sp_model(sent)

    spans = list(sent.ents) + list(sent.noun_chunks)  # collect nodes
    spans = spacy.util.filter_spans(spans)
    with sent.retokenize() as retokenizer:
        [retokenizer.merge(span, attrs={'tag': span.root.tag,
                                        'dep': span.root.dep}) for span in spans]
          
          
    # deps = [token.dep_ for token in sent]
    # cond =  (deps.count('obj') + deps.count('dobj')) != 1\
    #             or (deps.count('subj') + deps.count('nsubj')) != 1
    # if cond:
    #         continue

    for token in sent:
      if token.dep_ not in ('obj', 'dobj'):  # identify object nodes
          continue
      subject = [w for w in token.head.lefts if w.dep_
                  in ('subj', 'nsubj')]  # identify subject nodes
      if subject:
          subject = subject[0]
          # identify relationship by root dependency
          relation = [w for w in token.ancestors if w.dep_ == 'ROOT']
          if relation:
              relation = relation[0]
              # add adposition or particle to relationship
              if relation.i < len(sent) - 1 and relation.nbor(1).pos_ in ('ADP', 'PART'):
                  relation = ' '.join((str(relation), str(relation.nbor(1))))
          else:
              continue

          subject, subject_type = refine_ent(subject, sent, sp_model)
          obj, object_type = refine_ent(token, sent, sp_model)

          ent_pairs.append((str(subject), str(relation), str(obj), col))
    
    ent_pairs = [sublist for sublist in ent_pairs if not any(str(ent) == '' for ent in sublist)]
    filtered_triples = filter_triples(ent_pairs)
  return filtered_triples

""" Method 3 --> OMIT!!!! """
# verb_patterns = [[{"POS":"AUX"}, {"POS":"VERB"}, 
#                   {"POS":"ADP"}], 
#                  [{"POS":"AUX"}],
#                  [{"POS":"VERB"}]]
verb_patterns = [[{'POS': 'VERB', 'OP': '?'},
           {'POS': 'ADV', 'OP': '*'},
           {'POS': 'VERB', 'OP': '+'}], [{"POS":"VERB"}]]
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

def find_noun_phrase(verb_phrase, noun_phrases, side):
    for noun_phrase in noun_phrases:
        if (side == "left" and \
            noun_phrase.start < verb_phrase.start):
            return noun_phrase.text.strip()
        elif (side == "right" and \
              noun_phrase.start > verb_phrase.start):
            return noun_phrase.text.strip()
  
def find_triplet(doc, col, sp_model):
  triples = []
  for s in sp_model(doc).sents:
    sent = sp_model(s.text.strip())
    verb_phrases = get_verb_phrases(sent, sp_model)
    if len(verb_phrases) == 0:
      continue

    noun_phrases = sent.noun_chunks

    verb_phrase = None
    if (len(verb_phrases) > 1):
        verb_phrase = longer_verb_phrase(list(verb_phrases))
    else:
        verb_phrase = verb_phrases[0]

    left_noun_phrase = find_noun_phrase(verb_phrase, noun_phrases, "left")
    right_noun_phrase = find_noun_phrase(verb_phrase, noun_phrases, "right")
  
    triples.append((left_noun_phrase, verb_phrase.text, right_noun_phrase, col))

  filtered_triples = filter_triples(triples)

  return filtered_triples

""" Get relations and draw visualisations """
def get_data_triples(topics_to_docs, coref_intros, stopwords, function, *args):
  df_data = []
  colours= ['black', 'red', 'green', 'blue', 'yellow']
  topic_colouring = dict(zip(topics_to_docs.keys(), colours))

  for t, docs in topics_to_docs.items():
    rels = [function(remove_stopwords(clean_text(coref_intros[d]), stopwords), topic_colouring[t], *args) for d in docs]
    df_data.append(itertools.chain(*rels))

  df_data = itertools.chain(*df_data)
  
  if function.__name__ == "get_cooccurences":
    df_triples = pd.DataFrame(df_data, columns=['subject', 'objects', 'color'])
  else: 
    df_triples = pd.DataFrame(df_data, columns=['subject', 'relation', 'objects', 'color'])

  return df_triples

def draw_kg(pairs, method, file_suffix, show_rels=True):
  k_graph = nx.from_pandas_edgelist(pairs, 'subject', 'objects',create_using=nx.MultiDiGraph(), edge_attr='color')
  node_deg = nx.degree(k_graph)
  layout = nx.spring_layout(k_graph, k=1, iterations=60)
  plt.figure(figsize=(35,30))
  nx.draw_networkx(
      k_graph,
      node_size=[int(deg[1]) * 1000 for deg in node_deg],
      linewidths=1.5,
      pos=layout,
      edge_color=nx.get_edge_attributes(k_graph,'color').values(),
      edgecolors='black',
      node_color='white',
      )
  if show_rels:
    labels = dict(zip(list(zip(pairs.subject, pairs.objects)),pairs['relation'].tolist()))
    nx.draw_networkx_edge_labels(k_graph, pos=layout, edge_labels=labels,font_color='black')
  plt.savefig('./out/kg-{0}-{1}'.format(method, file_suffix))