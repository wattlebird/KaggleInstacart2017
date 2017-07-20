import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

data = "/mnt/d/Data/Instacart/"

cv = [
    [0,1], [0,2], [0,3], [0,4], [1,2], [1,3], [1,4], [2,3], [2,4], [3,4]
]

f = pd.read_hdf(data+"dataset.hdf", "train")

print("Features used:")
print(f.columns.values)

TH = 0.226

def f1(preds, train_data):
    Yt = train_data.get_label()
    return 'f1', f1_score(Yt, preds>TH), True

grpf1 = lambda x: pd.Series(data={
    'f1': 1.0
}) if not (x.label.any() or x.pred.any()) else pd.Series(data={
    'f1': f1_score(x.label, x.pred)
})

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'auc'},
    'num_leaves': 192,
    'max_depth': 11,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5, 
    'min_data_in_leaf': 1
}

min_data_in_leaf = [50, 100, 200, 400, 800, 1600]

for p in min_data_in_leaf:
    print("New iteration:")
    params['min_data_in_leaf'] = p

    print("Configuration: ")
    print(params)

    f1v = []
    aucv = []
    thv = []

    for i, val in enumerate(cv):
        print("Cross validation process iteration {0}".format(i+1))
        train = f[~f['seed'].isin(val)].drop(['order_id', 'user_id', 'product_id', 'seed'], axis=1)
        valid = f[f['seed'].isin(val)].drop(['order_id', 'user_id', 'product_id', 'seed'], axis=1)

        X = lgb.Dataset(train.drop('label', axis=1), train['label'], categorical_feature=['aisle_id', 'department_id', 'order_dow', 'order_hour_of_day', 'user_prod_reordered', 'user_prod_recentlydiscovered'])
        V = lgb.Dataset(valid.drop('label', axis=1), valid['label'], categorical_feature=['aisle_id', 'department_id', 'order_dow', 'order_hour_of_day', 'user_prod_reordered', 'user_prod_recentlydiscovered'], reference=X)
        evals_result = {}

        print("Training...")
        gbdt = lgb.train(params, X, valid_sets=V, verbose_eval=True)
        print("Training end for iteration {0}".format(i+1))

        gbdt.save_model("model_iter{0}.txt".format(i+1))

        yp = gbdt.predict(valid.drop('label', axis=1))
        thfinder = lambda x: -f1_score(valid.label.values, yp>x)
        th = minimize_scalar(thfinder, bracket=(0.15, 0.25), method='brent')
        f1v.append(f1_score(valid['label'].values, yp>th.x))
        aucv.append(roc_auc_score(valid['label'].values, yp))
        thv.append(th.x)
        print("F1: {0}\nAUC: {1}\nTH: {2}".format(f1_score(valid['label'].values, yp>th.x), roc_auc_score(valid['label'].values, yp), th.x))

        if i==0:
            print("Print DSAT debug case...")
            valid_label = f[['order_id', 'user_id', 'product_id', 'label']][f['seed'].isin(val)]
            valid_label['pred'] = np.require(yp>th.x, dtype=np.int)
            valid_label['pred_prob'] = yp
            f1s = valid_label[['user_id', 'label', 'pred', 'pred_prob']].groupby(by='user_id').apply(grpf1)
            valid_label = valid_label.merge(f1s, left_on='user_id', right_index=True).sort_values('f1')
            valid_label.to_csv("dsatdebug.tsv", sep='\t', index=False)
            del f1s, valid_label

        del X, V, gbdt

    print("Avg F1: {0}".format(np.mean(f1v)))
    print("Avg auc: {0}".format(np.mean(aucv)))
    print("Avg TH: {0}".format(np.mean(thv)))
