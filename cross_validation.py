import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
import lightgbm as lgb
import matplotlib.pyplot as plt

data = "/mnt/d/Data/Instacart/"

cv = [
    [0,1], [0,2], [0,3], [0,4], [1,2], [1,3], [1,4], [2,3], [2,4], [3,4]
]

f = pd.read_hdf(data+"dataset.hdf", "train")

print("Features used:")
print(f.columns.values)

TH = 0.20

def f1(preds, train_data):
    Yt = train_data.get_label()
    return 'f1', f1_score(Yt, preds>TH), True

print("Cutting threshold: {0}".format(TH))

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'auc'},
    'num_leaves': 127,
    'max_depth': 10,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5
}

print("Configuration: ")
print(params)

for i, val in enumerate(cv):
    print("Cross validation process iteration {0}".format(i+1))
    train = f[~f['seed'].isin(val)].drop(['order_id', 'user_id', 'product_id', 'seed'], axis=1)
    valid = f[f['seed'].isin(val)].drop(['order_id', 'user_id', 'product_id', 'seed'], axis=1)

    X = lgb.Dataset(train.drop('label', axis=1), train['label'], categorical_feature=['aisle_id', 'department_id', 'order_dow', 'order_hour_of_day', 'user_prod_reordered', 'user_prod_recentlydiscovered'])
    V = lgb.Dataset(valid.drop('label', axis=1), valid['label'], categorical_feature=['aisle_id', 'department_id', 'order_dow', 'order_hour_of_day', 'user_prod_reordered', 'user_prod_recentlydiscovered'], reference=X)
    evals_result = {}

    print("Training...")
    gbdt = lgb.train(params, X, valid_sets=V, feval=f1, verbose_eval=False)
    print("Training end for iteration {0}".format(i+1))

    ax = lgb.plot_importance(gbdt)
    plt.savefig("feature_importance_cv{0}.png".format(i+1))

    yp = gbdt.predict(valid.drop('label', axis=1))
    print("F1: {0}\nAUC: {1}".format(f1_score(valid['label'].values, yp>TH), roc_auc_score(valid['label'].values, yp)))
