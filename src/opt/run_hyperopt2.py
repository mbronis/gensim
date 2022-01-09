import pickle
from datetime import datetime
from collections import Counter

import pandas as pd
import numpy as np
import spacy
import optuna
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preproc import tokenize
from src.io import csv_loader_factory
from src.preprocessing import TextCleaner
from src.embeddings import W2VEmbedder


class PicklingCallback:
    def __init__(self, suffix: str):
        self.path = './opt_studies/test_study_' + suffix + '.pkl'

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        with open(self.path, 'wb') as p:
            pickle.dump(study, p)


def make_stopwords(texts: pd.Series, freq_tresh: float = 0.02):
    words = list(map(lambda str: str.split(), texts))
    words = [item for sublist in words for item in sublist]

    words_count = Counter(words)

    return set([k for k, v in words_count.items() if v > len(texts) * freq_tresh])

def lemmatize_pipe(doc, stopwords):
    lemma_list = [str(tok.lemma_).lower() for tok in doc
                  if tok.text.lower() not in stopwords
                 ] 
    return ' '.join(lemma_list)

def preprocess_pipe(texts, stopwords):
    if isinstance(texts, str): texts = [texts]
    preproc_pipe = []
    for doc in nlp.pipe(texts, batch_size=20):
        preproc_pipe.append(lemmatize_pipe(doc, stopwords))
    return pd.Series(preproc_pipe, index=texts.index)

def objective(trial):
    # clean
    cleaner_params = {
        'latinize': trial.suggest_categorical('latinize', [False, True]),
        'to_lower': trial.suggest_categorical('to_lower', [False, True]),
    }
    cleaner = TextCleaner(**cleaner_params)
    train_clean = cleaner.fit_transform(train_raw)
    test_clean = cleaner.transform(test_raw)

    # stop words
    freq_tresh = trial.suggest_discrete_uniform("stopwords_tresh", 0.001, 0.201, 0.005)
    stopwords = make_stopwords(train_clean, freq_tresh)

    # lemmatize
    train_prepro = preprocess_pipe(train_clean, stopwords)
    test_prepro = preprocess_pipe(test_clean, stopwords)

    # vectorize
    ngram_range_hi =  trial.suggest_int('ngram_range_hi', 1, 3, 1)
    vec_params = {
        'ngram_range': (1, ngram_range_hi)
    }
    vectorizer = TfidfVectorizer(**vec_params)
    train_vec = vectorizer.fit_transform(train_prepro)

    # select words with with chi2
    _, p = chi2(train_vec, train_label)
    df_selection = pd.DataFrame({'feature': vectorizer.get_feature_names_out(), 'score': 1-p})
    p_value_limit =  trial.suggest_discrete_uniform("selector_p_value_limit", 0.0, 0.95, 0.05),
    features_selected = df_selection[df_selection['score'] > p_value_limit]['feature']

    # vectorize on selected
    vec_params = {
        'ngram_range': (1, ngram_range_hi),
        'vocabulary': features_selected
    }
    vectorizer = TfidfVectorizer(**vec_params)
    vectorizer.fit(train_prepro)    
    train_vec = vectorizer.transform(train_prepro)
    test_vec = vectorizer.transform(test_prepro)

    
    # fit model
    cb_params = {
        'eval_metric': 'AUC',
        'early_stopping_rounds': 50,
        'verbose': 0,
        'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 101, 10),
        'learning_rate': trial.suggest_discrete_uniform("learning_rate", 0.001, 0.101, 0.005),
        'reg_lambda': trial.suggest_int('reg_lambda', 1, 9, 2),
        'max_depth': trial.suggest_int('max_depth', 2, 16, 1),
        'num_leaves': trial.suggest_int('num_leaves', 6, 36, 6),
    }

    cb = CatBoostClassifier(**cb_params)
    cb.fit(X=train_vec, y=train_label, eval_set=(test_vec, test_label))

    # eval model
    proba_train = cb.predict_proba(train_vec)[:, 1]
    proba_test = cb.predict_proba(test_vec)[:, 1]
    
    tresh = np.quantile(proba_train, np.mean(train_label))
    pred_test = (proba_test<tresh).astype(int)
    
    auc = roc_auc_score(test_label, proba_test)
    f1 = f1_score(test_label, pred_test)
    
    return auc, f1


if __name__ == '__main__':
    print('loading model')
    nlp = spacy.load('pl_core_news_md', disable=['parser', 'tagger', 'ner', 'attribute_ruler'])

    print('loading data')
    loader = csv_loader_factory('poleval')
    data = loader.load()

    train_raw = data.loc[data['dataset']=='train', 'text_raw']
    train_label = data.loc[data['dataset']=='train', 'tag']
    test_raw = data.loc[data['dataset']=='test', 'text_raw']
    test_label = data.loc[data['dataset']=='test', 'tag']

    suffix = datetime.now().strftime("%Y%m%d_%H%M")
    pkl_cb = PicklingCallback(suffix)

    study = optuna.create_study(directions=["maximize", "maximize"])
    study.optimize(objective, n_trials=5000, n_jobs=-1, catch=(Exception,), callbacks=[pkl_cb])
