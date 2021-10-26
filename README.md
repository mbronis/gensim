# `gensim` tutorial

This repo serves the purpose of exploring the `gensim` package and hone pipelines usage for processing data and tuning models.


Dataset used in this repo is polish language sentiment dataset (see data/info.txt). It consist of ~1M texts, 20% of which are labeled as negative.

The idea is to use gensim models to produce text embeddings and run catboost models on them.

# Embeddings
Idea of embedding is to create fixed size reprezentation (number of columns) of texts.  
Several approaches will be tested:
* bow with tfidf
* w2v -> centroid
* w2v -> clustering -> tfidf on count of w2v embedded words in each cluster
* d2v
* other supported by `gensim`

In later steps `spacy` model will be used for tokenization with lemmatizer.

# Pipelines
`sk-learn` pipelines allow for standarization of ML pipeline steps.  
It also allow for easy parametrization of each step, which can be usefull in optimization of parameters.  
