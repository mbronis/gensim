from typing import Tuple

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
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

    mean = pd.DataFrame(zip([round(np.mean(proba_train), 6)], [round(np.mean(proba_test), 6)]),
             columns=['train', 'test'], index=['Mean'])

    return pd.concat([res, mean])

def plot_preds(act, pred):
    assert len(act) == len(pred), f'Len of `act` and `pred` must match {len(act)} != {len(pred)}'

    ax = pd.Series(pred).plot.hist()
    ax.set_title('Preds hist')
    plt.show()

    df = pd.DataFrame({'act': act, 'pred': pred})
    df['pred_q'] = pd.qcut(df['pred'], q=[x/10 for x in range(11)], labels=False, duplicates='drop')


    dfg = df.groupby('pred_q').agg(
        count=('act', 'size'),
        act=('act', 'mean'),
        pred=('pred', 'mean')
    )

    ax1 = dfg[['count']].plot(kind='bar')
    ax2 = ax1.twinx()
    dfg[['act', 'pred']].plot(kind='line', ax=ax2, color=['tab:red', 'tab:green'])

    ax1.set_title('Act vs predicted')
    ax1.set_ylabel('Count')
    ax2.set_ylabel('Freq')

    plt.show()

    return dfg
