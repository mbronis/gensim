import pickle
from datetime import datetime

import numpy as np
import optuna
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, f1_score

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

def objective(trial):
    # clean
    cleaner_params = {
        'latinize': trial.suggest_categorical('latinize', [False, True]),
        'to_lower': trial.suggest_categorical('to_lower', [False, True]),
    }
    cleaner = TextCleaner(**cleaner_params)
    train_clean = cleaner.fit_transform(train_raw)
    test_clean = cleaner.transform(test_raw)
    
    # tokenize
    train_tokens, train_label_tok = tokenize(train_clean, train_label)
    test_tokens, test_label_tok = tokenize(test_clean, test_label)
    
    # vectorize
    emb_params = {
        'vector_size': trial.suggest_int('vector_size', 50, 1000, 50),
        'sg': trial.suggest_categorical('sg', [0, 1]),
        'max_final_vocab': trial.suggest_int('max_final_vocab', 500, 20000, 100),
        'min_count': trial.suggest_int('min_count', 5, 200, 5),
        'ns_exponent': trial.suggest_discrete_uniform("ns_exponent", -0.5, 1.0, 0.05),
        'alpha': trial.suggest_discrete_uniform("alpha", 0.001, 0.101, 0.005),
        'epochs': trial.suggest_int('epochs', 3, 10, 1),        
        'workers': 16,
    }
    emb = W2VEmbedder(**emb_params)
    emb.fit(train_tokens)
    
    train_w2v = emb.transform(train_tokens)
    test_w2v = emb.transform(test_tokens)
    
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
    cb.fit(X=train_w2v, y=train_label_tok, eval_set=(test_w2v, test_label_tok))

    # eval model
    proba_train = cb.predict_proba(train_w2v)[:, 1]
    proba_test = cb.predict_proba(test_w2v)[:, 1]
    
    tresh = np.quantile(proba_train, np.mean(train_label_tok))
    pred_test = (proba_test<tresh).astype(int)
    
    auc = roc_auc_score(test_label_tok, proba_test)
    f1 = f1_score(test_label_tok, pred_test)
    
    return auc, f1


if __name__ == '__main__':
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
