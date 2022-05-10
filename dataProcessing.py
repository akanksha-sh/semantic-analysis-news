from itertools import chain
import allennlp
import pandas as pd
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
import re

def get_intros(articles, sp):
  """Get first 5 sentences from each article """
  intro_sents = [list(map(lambda x: x.text, sp(a).sents))[:5] for a in articles]
  intros = list(map(' '.join, intro_sents))
  return intros

def clean_text(text):
  """Remove multiple spaces in content"""
  ctext = re.sub(r"\s+", " ", text)
  """Remove ellipsis"""
  ctext = re.sub(r"…", "", ctext)
  """Replace dash between words"""
  ctext = re.sub(r"(?<=\w)-(?=\w)| --", " ", ctext)
  # """Replace punctuation and unwanted chars"""
  # ctext = re.sub(r'[,!@#$%^&*)(|/><";:.?\'\“\”\'\\}{]', '', ctext)

  # text = re.sub(
  #       f"[{re.escape(string.punctuation)}]", "", text
  return ctext

def filter_lemmatise_tokens(tokens, stopwords):
  filtered_tokens = []
  allowed_postags=['NOUN', 'ADJ']
  for token in tokens:
    w = token.text
    if token.pos_ not in allowed_postags or len(w) <= 1 or w in stopwords or w.isdigit():
      continue
    filtered_tokens.append(token.lemma_)
  return filtered_tokens

def get_filtered_tokens(doc, sp, stopwords):
    filtered_tokens = [filter_lemmatise_tokens(sp(s.text.strip()), stopwords) for s in sp(doc).sents]
    return list(chain(*filtered_tokens))