import urllib.request
import os

def load_models():
    urllib.request.urlretrieve("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz", "./models/coref-spanbert-large.tar.gz")
    print("coref laoded")
    
    urllib.request.urlretrieve("https://storage.googleapis.com/allennlp-public-models/fine-grained-ner.2021-02-11.tar.gz", "./models/fine-grained-ner.tar.gz")
    print("ner loaded")
    
    urllib.request.urlretrieve("https://storage.googleapis.com/allennlp-public-models/stanford-sentiment-treebank-roberta.2021-03-11.tar.gz", "./models/roberta-sentiment.tar.gz")
    print("sentiment loaded")

if __name__ == '__main__':
    os.makedirs("./models/")
    load_models()