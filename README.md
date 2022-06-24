# Semantic Analysis Engine For News

<div id="top"></div>

<!-- TABLE OF CONTENTS -->
  ## Table of Contents
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Libraries Used</a></li>
        <li><a href="#models">Pre-trained Models</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#output-files">Output Files</a></li>
    <li><a href="#visualisation-tool">Visualisation Tool</a></li>
     <li><a href="#license">License</a></li>
  </ol>


<!-- ABOUT THE PROJECT -->
## About The Project

This repository presents a solution for information retrieval of online news articles. It provides an end-to-end pipeline which focuses on two main aspects of semantic analysis: topic extraction and knowledge representation through semantic triples. It takes an input dataset (in our case, airline.csv) and uses a range a natural language processes to output the retrieved information such as a 'semantic cluster' &rarr; 'latent topics' &rarr; articles mapping along with information like topic name, keywords, topic sentiment etc. and a collection of semantic triples of type subject &rarr; predicate &rarr; object along with information such as triple sentiment and associated topic and article. 

For a more detailed insight into the key motivations and findings for this project, please refer to the following associated paper: <a id="raw-url" href="https://github.com/akanksha-sh/FYP_report/blob/main/main.pdf">Download here</a>

<p align="right">(<a href="#top">back to top</a>)</p>

### Built With

This project made use of several NLP libraries and frameworks which include:

* [AllenNLP](https://allenai.org/allennlp/software/allennlp-library)
* [AllenNLP Models](https://github.com/allenai/allennlp-models/)
* [SpaCy](https://https://spacy.io/)
* [Gensim](https://github.com/RaRe-Technologies/gensim)

### Pre-trained Models

This project made use of several NLP libraries and frameworks which include:

* [SpanBERT Coreference Resolution](https://github.com/allenai/allennlp-models/blob/main/allennlp_models/modelcards/coref-spanbert.json) - AllenNLP Models

* [Fine Grained Named Entity Recognition](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/modelcards/tagging-fine-grained-crf-tagger.json) - AllenNLP Models

* [RoBERTa Large for SST](https://github.com/allenai/allennlp-models/blob/main/allennlp_models/modelcards/roberta-sst.json) - AllenNLP Models

* [Latent Dirichlet Allocation](https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/ldamodel.py) - Gensim

* [Wikipedia and Gigaword 5 GloVe embeddings](https://github.com/RaRe-Technologies/gensim/blob/2feef89a24c222e4e0fc6e32ac7c6added752c26/docs/src/gallery/howtos/run_downloader_api.py) - Gensim

* [English Language Model](https://spacy.io/models/en) - spaCy

<p align="right">(<a href="#top">back to top</a>)</p>


## Getting Started

### Prerequisites

Requires access to a **GPU that supports CUDA**. 
```
ssh <gpu_machine>
```

### Installation

Start a new **VIRTUAL ENVIRONMENT** and install all the necessary dependencies as from requirements.txt.

```
python3 -m venv venv
. venv/bin/activate
pip3 install -r requirements.txt
```

Clone this repository in the virtual environment.


```
git clone https://github.com/akanksha-sh/semantic-analysis-news.git
```

## Usage

To submit a job on the GPU:
```
sbatch job_test.sh
```

To run the pipeline locally on the GPU:
```
cd src/
python3 visualisingnewsv2.py
```
<p align="right">(<a href="#top">back to top</a>)</p>

## Output Files

Once the pipeline has finished running, the engine will dump the output results in the <tt> **data/** </tt> folder. This folder contains the following files: 

* <tt>json/</tt> folder: contains all the semantic triples for each Year-Category input group

* <tt>combined_articles.csv</tt>: all the article information containing title, categoty, year published, sentiment, associated topic etc.

* <tt>combined_topics.csv</tt>: all the topic information containing inferred topic name, keywords, sentiment, associated articles etc.

<p align="right">(<a href="#top">back to top</a>)</p>

## Visualisation Tool

In order to display the results of this semantic analysis engine, this repository can be used in conjunction with the Visualisation Tool from this repository: <a href=https://github.com/akanksha-sh/visualisation-semantic-analysis.git> <tt> visualisation-semantic-analysis </tt> </a>

<p align="right">(<a href="#top">back to top</a>)</p>

## License
Copyright &#169; 2022 Akanksha Sharma

Licensed under the MIT license. 