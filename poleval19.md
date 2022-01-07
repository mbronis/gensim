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