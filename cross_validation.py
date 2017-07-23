import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.optimize import minimize_scalar
from parametertunning import gbdt_cross_validation_data, gbdt_training, mailsend, mailsendfail
from setting import *


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

                th = minimize_scalar(obj, bracket=(0.15, 0.2), method='brent', tol=1e-2)
                print("Threshold found. {0}.".format(th.x))

                f1temp.append(valid_label[['user_id', 'label', 'pred_prob']].\
                        groupby('user_id').\
                        apply(lambda x: f1_score(x.label, x.pred_prob>th.x)).\
                        mean())
                auctemp.append(roc_auc_score(valid['label'].values, Yp))
            
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