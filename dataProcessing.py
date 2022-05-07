import pandas as pd
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
import re


data = pd.read_csv('airline.csv', parse_dates=[1])
data.columns = ['url', 'date', 'title', 'author', 'category', 'article']

def get_group_data(category_key, time_key, time_freq):
  categories_df = data.groupby('category')
  cat_key = category_key
  cat_group = categories_df.get_group(cat_key)
  cat_group.reset_index(drop=True, inplace=True)
  time_df = cat_group.groupby(pd.Grouper(key='date', freq=time_freq))
  time_key = time_key
  group = time_df.get_group(time_key)
  group.reset_index(drop=True, inplace=True)
  return group

def get_intros(articles):
  """Get first 5 sentences from each article """
  sentsplitter = SpacySentenceSplitter() 
  all_sents = [sentsplitter.split_sentences(a) for a in articles] 
  intro_sents = [a[:5] for a in all_sents]
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

""" Tokenise and lemmatise """
def filter_lemmatise_tokens(tokens, stopwords):
  filtered_tokens = []
  allowed_postags=['NOUN', 'VERB']

  for token in tokens:
    w = token.text
    if token.pos_ not in allowed_postags or len(w) <= 1 or w in stopwords or w.isdigit():
      continue
    filtered_tokens.append(token.lemma_)
  return filtered_tokens

def get_filtered_tokens(docs, stopwords):
    tokeniser = SpacyTokenizer()
    tokens_list = [tokeniser.tokenize(d) for d in docs]
    filtered_tokens = [filter_lemmatise_tokens(tokens, stopwords) for tokens in tokens_list]
    return filtered_tokens

