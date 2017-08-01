import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
import lightgbm as lgb
from setting import *
from parametertunning import gbdt_get_training_data, gbdt_get_testing_data, gbdt_training
from datetime import datetime
import gc

gc.enable()
params = gbdt_params

def main():
    train = gbdt_get_training_data()

    gbdt = gbdt_training(params, train.drop(['order_id', 'user_id', 'product_id', 'seed'], axis=1))
    gbdt.save_model("/tmp/model.txt")
    uploadfile("/tmp/model.txt", "model.txt")
    test = gbdt_get_testing_data()
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
    uploadfile("result.csv", "result_{0}_filltrain.csv".format(datetime.today().strftime("%Y-%m-%d_%H_%M_%S")))

if __name__=="__main__":
    main()
