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

grpf1 = lambda x: pd.Series(data={
    'f1': 1.0
}) if not (x.label.any() or x.pred.any()) else pd.Series(data={
    'f1': f1_score(x.label, x.pred)
})

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

f1v = []
aucv = []

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

    gbdt.save_model("model_iter{0}.txt".format(i+1))

    yp = gbdt.predict(valid.drop('label', axis=1))
    f1v.append(f1_score(valid['label'].values, yp>TH))
    aucv.append(roc_auc_score(valid['label'].values, yp))
    print("F1: {0}\nAUC: {1}".format(f1_score(valid['label'].values, yp>TH), roc_auc_score(valid['label'].values, yp)))

    if i==0:
        print("Print DSAT debug case...")
        valid_label = f[['order_id', 'user_id', 'product_id', 'label']][f['seed'].isin(val)]
        valid_label['pred'] = np.require(yp>TH, dtype=np.int)
        valid_label['pred_prob'] = yp
        f1 = valid_label[['user_id', 'label', 'pred', 'pred_prob']].groupby(by='user_id').apply(grpf1)
        valid_label = valid_label.merge(f1, left_on='user_id', right_index=True).sort_values('f1')
        valid_label.to_csv("dsatdebug.tsv", sep='\t', index=False)

    del X, V, gbdt

print("Avg F1: {0}".format(np.mean(f1v)))
print("Avg auc: {0}".format(np.mean(aucv)))