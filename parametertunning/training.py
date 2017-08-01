import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

def gbdt_training(param, train, valid=None, verbose=True, keep_training_booster=False):
    X = lgb.Dataset(train.drop('label', axis=1), train['label'])
    if valid is not None:
        V = lgb.Dataset(valid.drop('label', axis=1), valid['label'], reference=X)
        gbdt = lgb.train(param, X, valid_sets=[X, V], num_boost_round=200, early_stopping_rounds=20, verbose_eval=verbose, keep_training_booster=False)
        return gbdt

    gbdt = lgb.train(param, X, valid_sets=[X], num_boost_round=200, early_stopping_rounds=20, verbose_eval=verbose, keep_training_booster=False)

    return gbdt
