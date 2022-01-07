# PolEval19 learning

## data
**Task**: binary classification  
**Features**: pl tweet texts


|dataset|size|freq|
|-------|----|----|		
test	|999	|0.134134|
train	|10040	|0.084761|


## Results
----
### catboost on raw text
| metric    | train      | test   |
|-----------|------------|--------|
AUC			|0.560178	|0.518635|
Accuracy	|0.834761	|0.802803|
F1			|0.023543	|0.075117|
Precision	|0.023585	|0.101266|
Recall		|0.023502	|0.059701|
|mean pred |0.158266|0.158236|

Low variance of preds.

----
### catboost on cleaned text
Cleaner params:
```python
'clean_email': True,
'clean_emoji': True,
'clean_hashtag': True,
'clean_non_alpha': True,
'clean_non_letter': True,
'clean_url': True,
'clean_user_ref': True,
'drop_repeated': True,
'latinize': True,
'to_lower': True
```
| metric    | train      | test   |
|-----------|------------|--------|
AUC			|0.526524	|0.509443|
Accuracy	|0.858765	|0.823824|
F1			|0.041892	|0.073684|
Precision	|0.049285	|0.125000|
Recall		|0.036428	|0.052239|
|mean pred |0.24648|0.246439|

Even lower preds variance.  
Suprisingly high average pred. To be investigated.

----
### catboost w2v embedding
Embedder params:
```python
'alpha': 0.025,
'cbow_mean': 1,
'epochs': 5,
'hs': 0,
'max_final_vocab': None,
'min_alpha': 0.0001,
'min_count': 5,
'model_name': 'w2v',
'negative': 5,
'ns_exponent': 0.75,
'sample': 0.001,
'sg': 0,
'vector_size': 20,
'window': 5,
'workers': 8
```
| metric    | train      | test   |
|-----------|------------|--------|
AUC			|0.551923	|0.515249|
Accuracy	|0.859565	|0.819820|
F1			|0.063830	|0.072165|
Precision	|0.073282	|0.116667|
Recall		|0.056537	|0.052239|
|mean pred |0.379661|0.379834|

----
### svm on w2v

| metric    | train      | test   |
|-----------|------------|--------|
AUC			|0.500769	|0.518670|
Accuracy	|0.846000	|0.793794|
F1			|0.090695	|0.096491|
Precision	|0.090695	|0.117021|
Recall		|0.090695	|0.082090|
|mean pred |0.083677|0.083493|

* best model so far, still very poor

----
### cb w2v 100

| metric    | train      | test   |
|-----------|------------|--------|
AUC			|0.785252	|0.511988|
Accuracy	|0.831039	|0.802803|
F1			|0.002356	|0.075117|
Precision	|0.002356	|0.101266|
Recall		|0.002356	|0.059701|
|mean pred |0.083908|0.085098|

* metrics decreased slightly
* mean pred matches act freq
* distribution of preds with normal shape

----
### svm w2v 100

| metric    | train      | test   |
|-----------|------------|--------|
AUC			|0.512709	|0.482340|
Accuracy	|0.842210	|0.805806|
F1			|0.068316	|0.126126|
Precision	|0.068316	|0.159091|
Recall		|0.068316	|0.104478|
|mean pred |0.083236|0.083327|

* auc detoriated below 0.5
* f1 improved

----
### cb w2v 500

| metric    | train      | test   |
|-----------|------------|--------|
AUC			|0.670866	|0.473311|
Accuracy	|0.835228	|0.810811|
F1			|0.027091	|0.078049|
Precision	|0.027091	|0.112676|
Recall		|0.027091	|0.059701|
|mean pred |0.086777|0.0878|

* increasing vector size above 100 leads to overfitting (at least with default cb params)


----
### man tuning w2v and cb

```python
# w2v
'alpha': 0.025,
'cbow_mean': 1,
'epochs': 10,
'hs': 0,
'max_final_vocab': None,
'min_alpha': 0.0001,
'min_count': 5,
'model_name': 'w2v',
'negative': 5,
'ns_exponent': 0.75,
'sample': 0.001,
'sg': 1,
'vector_size': 250,
'window': 5,
'workers': 8

# cb
'early_stopping_rounds': 50,
'grow_policy': 'Depthwise',
'min_data_in_leaf': 10,
'learning_rate': 0.01,
'reg_lambda': 5
```

| metric    | train      | test   |
|-----------|------------|--------|
AUC			|0.625901	|0.564416|
Accuracy	|0.835428	|0.807808|
F1			|0.028269	|0.085714|
Precision	|0.028269	|0.118421|
Recall		|0.028269	|0.067164|
|mean pred |0.397805|0.397815|

* slightly improved results
* mean pred failed
