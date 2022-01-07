# PolEval19 learning

## data
Task: binary classification
Datasize: 10k
Target freq: 10%
Features: tweet texts


## results

### catboost on raw text
	        train	    test
AUC	        0.560178	0.518635
Accuracy	0.915239	0.865866
F1	        0.000000	0.000000
Precision	1.000000	1.000000
Recall	    0.000000	0.000000

### catboost on cleaned text
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

			train		test
AUC			0.526524	0.509443
Accuracy	0.915239	0.865866
F1			0.000000	0.000000
Precision	1.000000	1.000000
Recall		0.000000	0.000000