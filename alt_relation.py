""" Method 1 - entity triples"""
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

def extract_relations(doc, col, sp_model):
  ent_pairs = []
  for s in sp_model(doc).sents:
    sent = sp_model(s.text.strip())

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
