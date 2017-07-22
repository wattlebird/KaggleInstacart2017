import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
import lightgbm as lgb
from parametertunning import gbdt_cross_validation_data, gbdt_training

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'auc'},
    'num_leaves': 192,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5
}

def main():
    thv = []
    modelv = []
    for i, (train, valid, valid_label) in enumerate(gbdt_cross_validation_data()):
        print("New iteration {0}, parameters: {1}".format(i+1, params))
        print("Start training...")
        model = gbdt_training(params, train, valid)
        print("Training finished.\nStart predicting...")
        
        Yp = model.predict(valid.drop('label', axis=1), num_iteration=model.best_iteration)
        valid_label['pred_prob'] = Yp
        print("Prediction finished.\nFinding optimal threshold...")

        def obj(y):
            return -valid_label[['user_id', 'label', 'pred_prob']].groupby('user_id').apply(lambda x: f1_score(x.label, x.pred_prob>y)).mean()

        th = minimize_scalar(obj, bracket=(0.15, 0.25), method='brent', tol=1e-5)
        print("Threshold found. {0}.".format(th.x))
        thv.append(th.x)
        modelv.append(model)

    test = pd.read_hdf("/mnt/d/Data/Instacart/dataset.hdf", "test")
    test['aisle_id'] = test.aisle_id.astype('category')
    test['department_id'] = test.department_id.astype('category')
    test['order_dow'] = test.order_dow.astype('category')
    test['order_hour_of_day'] = test.order_hour_of_day.astype('category')
    test['user_prod_reordered'] = test.user_prod_reordered.astype('category')
    test['user_prod_recentlydiscovered'] = test.user_prod_recentlydiscovered.astype('category')

    res = []
    for x, model in zip(thv, modelv):

