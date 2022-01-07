# `gensim` tutorial

This repo serves the purpose of exploring the `gensim` package and hone pipelines usage for processing data and tuning models.


Two dataset are used in this repo:
* polish language sentiment dataset. It consist of ~1M texts, 20% of which are labeled as `negative`.
* `kaggle` tweets dataset, with 1.6M tweets with two, evenly distributed labels: `neutral` and `positive`  
 see data/info.txt for details

The idea is to use gensim models to produce text embeddings and run `catboost` models on them.

# Embeddings
Idea of embedding is to create fixed size reprezentation (number of columns) of texts.  
Several approaches will be tested:
* bow with tfidf
* w2v -> centroid
* w2v -> clustering -> tfidf on count of w2v embedded words in each cluster
* d2v
* other supported by `gensim`

In later steps `spacy` model will be used for lemtaization of polish language words.

# Pipelines
`sk-learn` pipelines allow for standarization of ML pipeline steps.  
It also allow for easy parametrization of each step, which can be usefull in optimization of parameters.  


# Text features engineering
After some eda I have some ideas for new features other than embedings described above. Ideas for this features arrises from questions about relation of data and sentiment.  

**based on user**:
* number of tweets made  
*if you tweet a lot you have particular sentiment?*
* average target of tweets  
*you are consistent in sentiment you tweet?*  
However this feature can leak data. Its impact on model performance will be tested with cross-validation with split based on users (all tweets of particular user should fall into the one fold). 

**user mentioned in tweet text**
* average of targets of tweets where user is mentioned  
*particular users, when refered in tweet have specific sentiment?*
* average of targets for tweets made by users mentioned  
*you use similar sentiment as users you mention?*
* remove/replace with dedicated token
* count

**links/emails/hashtags in text**
* remove/replace with dedicated token
* count
* hashtags:
    * create dedicated embeddings
    * remove #sign and threat as any other word




