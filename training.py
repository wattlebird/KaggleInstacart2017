import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
import lightgbm as lgb

data = "/mnt/d/Data/Instacart/"

train = pd.read_hdf("/mnt/d/Data/Instacart/dataset.hdf", "train").drop(['order_id', 'user_id', 'product_id', 'seed'], axis=1)
train['aisle_id'] = train.aisle_id.astype('category')
train['department_id'] = train.department_id.astype('category')
train['order_dow'] = train.order_dow.astype('category')
train['order_hour_of_day'] = train.order_hour_of_day.astype('category')
train['user_prod_reordered'] = train.user_prod_reordered.astype('category')
train['user_prod_recentlydiscovered'] = train.user_prod_recentlydiscovered.astype('category')

TH = 0.20

def f1(preds, train_data):
    Yt = train_data.get_label()
    return 'f1', f1_score(Yt, preds>TH), True

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'auc'},
    'num_leaves': 127,
    'max_depth': 10,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5, 
    'verbose': 1
}

X = lgb.Dataset(train.drop('label', axis=1), train['label'])

gbdt = lgb.train(params, X, num_boost_round=160)

gbdt.save_model("model.txt")

test = pd.read_hdf("/mnt/d/Data/Instacart/dataset.hdf", "test")
test['aisle_id'] = test.aisle_id.astype('category')
test['department_id'] = test.department_id.astype('category')
test['order_dow'] = test.order_dow.astype('category')
test['order_hour_of_day'] = test.order_hour_of_day.astype('category')
test['user_prod_reordered'] = test.user_prod_reordered.astype('category')
test['user_prod_recentlydiscovered'] = test.user_prod_recentlydiscovered.astype('category')
Y = gbdt.predict(test.drop(['order_id', 'user_id', 'product_id'], axis=1))

test['label'] = Y>TH

d = dict()
for row in test.itertuples():
    if row.label:
        try:
            d[row.order_id] += ' ' + str(row.product_id)
        except:
            d[row.order_id] = str(row.product_id)

for order in test.order_id:
    if order not in d:
        d[order] = 'None'
sub = pd.DataFrame.from_dict(d, orient='index')

sub.reset_index(inplace=True)
sub.columns = ['order_id', 'products']
sub.to_csv('result.csv', index=False)