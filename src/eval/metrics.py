from typing import Tuple

import pandas as pd
import numpy as np

from sklearn.metrics import auc, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

METRICS = {
    'AUC': roc_auc_score, 
    'Accuracy': accuracy_score,
    'F1': f1_score,
    'Precision': precision_score,
    'Recall': recall_score
}

def compute_metric(train_data: Tuple, test_data: Tuple, metric):
    
    act_train, proba_train, pred_train = train_data
    act_test, proba_test, pred_test = test_data
    
    if metric == 'AUC':
        train_score = METRICS[metric](act_train, proba_train)
        test_score = METRICS[metric](act_test, proba_test)
    else:
        train_score = METRICS[metric](act_train, pred_train)
        test_score = METRICS[metric](act_test, pred_test)
    
    df = pd.DataFrame(zip([train_score], [test_score]), columns=['train', 'test'], index=[metric])
    
    return df

def compute_metrics(train_data: Tuple, test_data: Tuple, metrics = METRICS):
    
    act_train, proba_train = train_data
    act_test, proba_test = test_data
    
    tresh = np.quantile(proba_train, np.mean(act_train))
    pred_train = (proba_train<tresh).astype(int)
    pred_test = (proba_test<tresh).astype(int)
    
    
    res = pd.DataFrame(columns=['train', 'test'])    
    for m in metrics:
        res_m = compute_metric((act_train, proba_train, pred_train), (act_test, proba_test, pred_test), m)
        res = pd.concat([res, res_m])
    
    return res
