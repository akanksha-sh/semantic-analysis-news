from itertools import chain
import re


def get_intros(titles, articles, sp, n=8):
    """Get first 8 sentences from each article"""
    intro_sents = [
        [titles[i] + "."] + list(map(lambda x: x.text, sp(a).sents))[:n]
        for i, a in enumerate(articles)
    ]
    intros = list(map(" ".join, intro_sents))
    return intros


def clean_text(text, unwanted_phrases={"Read More", "Source"}):
    """Clean text by removing elipsis, dash between words, unwanted characters and phrases"""
    ctext = re.sub(r"…", "", text)
    ctext = re.sub(r"(?<=\w)-(?=\w)| --", " ", ctext)
    ctext = re.sub(r"[@#^&*)(:|/><}{]", " ", ctext)
    ctext = re.sub(r"\s+", " ", ctext)
    pattern = re.compile(r"\b(" + r"|".join(unwanted_phrases) + r")\b\s*")
    ctext = pattern.sub(" ", ctext)
    return ctext


def get_entities(result, ignore_types=[]):
    """Named Entity Recognition using the BILOU tagging scheme"""
    entities = set()
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


def remove_entities(ner_pred, sent):
    ents = get_entities(ner_pred.predict(sent))
    pattern = re.compile(r"\b(" + r"|".join(ents) + r")\b\s*")
    text = pattern.sub(" ", sent)
    return text


def filter_lemmatise_tokens(tokens, stopwords):
    filtered_tokens = []
    allowed_postags = ["NOUN"]
    for token in tokens:
        w = token.text
        # POS tag filtering (i.e. Noun only corpus) or stopword removal
        if (
            token.pos_ not in allowed_postags
            or len(w) <= 1
            or token.lemma_ in stopwords
            or w.isdigit()
        ):
            continue
        filtered_tokens.append(token.lemma_)
    return filtered_tokens


def get_filtered_tokens(doc, sp, updated_stopwords):
    filtered_tokens = [
        filter_lemmatise_tokens(sp(s.text.strip()), updated_stopwords)
        for s in sp(doc).sents
    ]
    return list(chain(*filtered_tokens))
