import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.externals.joblib import Parallel, delayed
import lightgbm as lgb
from scipy.optimize import minimize_scalar
from parametertunning import gbdt_cross_validation_data, gbdt_training
from datetime import datetime
from setting import *

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
            return -np.mean(Parallel(n_jobs=4)(delayed(uf1score)(group, y) for _, group in valid_label[['user_id', 'label', 'pred_prob']].groupby('user_id')))

        th = minimize_scalar(obj, bracket=(0.15, 0.25), method='brent', tol=1e-5)
        print("Threshold found. {0}.".format(th.x))
        thv.append(th.x)
        modelv.append(model)

        if i==0:
            print("Dumping dsat debug file...")
            valid_label['pred'] = np.require(valid_label.pred_prob>th.x, dtype = np.int)
            prec = lambda g: precision_score(valid_label.ix[g.index]['label'].values, valid_label.ix[g.index]['pred'].values)
            reca = lambda g: recall_score(valid_label.ix[g.index]['label'].values, valid_label.ix[g.index]['pred'].values)
            accu = lambda g: accuracy_score(valid_label.ix[g.index]['label'].values, valid_label.ix[g.index]['pred'].values)
            f1sc = lambda g: f1_score(valid_label.ix[g.index]['label'].values, valid_label.ix[g.index]['pred'].values)

            debug_table = valid_label[['user_id', 'label', 'pred']].groupby(by='user_id').agg({'pred': {'precision': prec, 'recall': reca, 'accuracy': accu, 'f1': f1sc}})
            debug_table.to_csv("debug_{0}.tsv".format(i), sep='\t')
            valid_label.to_csv("debug_product_record_{0}.tsv".format(i), sep='\t')
            # since the file is very big one may upload it to azure 
            uploadfile("debug_{0}.tsv".format(i), "debug_{0}.tsv".format(feature_name))
            uploadfile("debug_product_record_{0}.tsv".format(i), "debug_product_record_{0}.tsv".format(feature_name))

    print("Loading test data...")
    test = pd.read_hdf(data+"dataset.hdf", "test")
    test['aisle_id'] = test.aisle_id.astype('category')
    test['department_id'] = test.department_id.astype('category')
    test['order_dow'] = test.order_dow.astype('category')
    test['order_hour_of_day'] = test.order_hour_of_day.astype('category')
    test['user_prod_reordered'] = test.user_prod_reordered.astype('category')
    test['user_prod_recentlydiscovered'] = test.user_prod_recentlydiscovered.astype('category')

    res = []
    for i, (x, model) in enumerate(zip(thv, modelv)):
        print("Predicting using model {0}".format(i+1))
        res.append(np.require(model.predict(test.drop(['order_id', 'user_id', 'product_id'], axis=1), num_iteration = model.best_iteration)>x, dtype=np.int))
        
    res = np.sum(np.vstack(res), axis=0)
    test['label'] = np.require(res>(len(modelv)//2 + 1), dtype=np.int)

    print("Generating result.tsv")
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
    uploadfile("result.csv", "result_{0}.csv".format(datetime.today().strftime("%Y-%m-%d_%H_%M_%S")))

if __name__=="__main__":
    main()
