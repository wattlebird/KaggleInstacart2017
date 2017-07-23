import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.optimize import minimize_scalar
from parametertunning import gbdt_cross_validation_data, gbdt_training, mailsend, mailsendfail
import os
from azure.storage.blob import BlockBlobService, ContentSettings


params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'auc'},
    'num_leaves': 224,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5,
    'verbose': 0
}

feature_name = "min_data_in_leaf"
title = "GDBT min_data_in_leaf tunning experiment"
candidate_param = [50, 100, 200, 400, 800]
block_blob_service = BlockBlobService(account_name=os.environ['AZURE_STORAGE_IKELY_ACCOUNT'], 
                                      account_key=os.environ['AZURE_STORAGE_IKELY_KEY'])


def main():
    
    try:
        f1v = []
        aucv = []
        print("Cross validation start.")
        for cp in candidate_param:
            params[feature_name] = cp
            f1temp = []
            auctemp = []
            print("{0} parameters to be tunned.".format(len(candidate_param)))

        
            for i, (train, valid, valid_label) in enumerate(gbdt_cross_validation_data()):
                print("New iteration {0}, parameters: {1}".format(i+1, params))
                print("Start training...")
                model = gbdt_training(params, train, valid, verbose=False)
                print("Training finished.\nStart predicting...")
                
                Yp = model.predict(valid.drop('label', axis=1), num_iteration=model.best_iteration)
                valid_label['pred_prob'] = Yp
                print("Prediction finished.\nFinding optimal threshold...")

                def obj(y):
                    return -valid_label[['user_id', 'label', 'pred_prob']].groupby('user_id').apply(lambda x: f1_score(x.label, x.pred_prob>y)).mean()

                th = minimize_scalar(obj, bracket=(0.15, 0.25), method='brent', tol=1e-5)
                print("Threshold found. {0}.".format(th.x))

                f1temp.append(valid_label[['user_id', 'label', 'pred_prob']].\
                        groupby('user_id').\
                        apply(lambda x: f1_score(x.label, x.pred_prob>th.x)).\
                        mean())
                auctemp.append(roc_auc_score(valid['label'].values, Yp))

                # if need to debug
                #if i==0:
                #    print("Dumping dsat debug file...")
                #    valid_label['pred'] = np.require(valid_label.pred_prob>th.x, dtype = np.int)
                #    prec = lambda g: precision_score(valid_label.ix[g.index]['label'].values, valid_label.ix[g.index]['pred'].values)
                #    reca = lambda g: recall_score(valid_label.ix[g.index]['label'].values, valid_label.ix[g.index]['pred'].values)
                #    accu = lambda g: accuracy_score(valid_label.ix[g.index]['label'].values, valid_label.ix[g.index]['pred'].values)
                #    f1sc = lambda g: f1_score(valid_label.ix[g.index]['label'].values, valid_label.ix[g.index]['pred'].values)

                #    debug_table = valid_label[['user_id', 'label', 'pred']].groupby(by='user_id').agg({'pred': {'precision': prec, 'recall': reca, 'accuracy': accu, 'f1': f1sc}})
                #    debug_table.to_csv("debug_{0}.tsv".format(i), sep='\t')
                #    valid_label.to_csv("debug_product_record_{0}.tsv".format(i), sep='\t')
                    # since the file is very big one may upload it to azure 
                #    block_blob_service.create_blob_from_path('temp', "debug_{0}.tsv".format(feature_name), "debug_{0}.tsv".format(i),
                #                                             content_settings=ContentSettings(content_type='text/tsv'))
                #    block_blob_service.create_blob_from_path('temp', "debug_product_record_{0}.tsv".format(feature_name), 
                #                                             "debug_product_record_{0}.tsv".format(i),
                #                                             content_settings=ContentSettings(content_type='text/tsv'))
            
            f1v.append(np.mean(f1temp))
            aucv.append(np.mean(auctemp))
            print("When parameter {0} set to {1}, we have average F1score {2} and average AUC {3}.\n\n".format(feature_name, cp,
                                                                                                           np.mean(f1temp),
                                                                                                           np.mean(auctemp)))
        mailsend(title, params, feature_name, candidate_param, f1v, aucv)

    except Exception as e:
        try:
            mailsendfail(title, params, e)
        except Exception as e2:
            raise e

if __name__=="__main__":
    main()
