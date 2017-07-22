import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.optimize import minimize_scalar
from parametertunning import gbdt_cross_validation_data, gbdt_training, mailsend, mailsendfail


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

a = [92, 128, 160, 192, 224, 256]

def main():
    title = "GDBT num_leaves tunning experiment"
    try:
        f1v = []
        aucv = []
        print("Cross validation start.")
        for num_leaves in a:
            params['num_leaves'] = num_leaves
            f1temp = []
            auctemp = []
            print("{0} parameters to be tunned.".format(len(a)))

        
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

                f1temp.append(valid_label[['user_id', 'label', 'pred_prob']].\
                        groupby('user_id').\
                        apply(lambda x: f1_score(x.label, x.pred_prob>th.x)).\
                        mean())
                auctemp.append(roc_auc_score(valid['label'].values, Yp))

                # if need to debug
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

            
            f1v.append(np.mean(f1temp))
            aucv.append(np.mean(auctemp))
        mailsend(title, params, 'num_leaves', a, f1v, aucv)

    except Exception as e:
        try:
            mailsendfail(title, params, e)
        except Exception as e2:
            raise e

if __name__=="__main__":
    main()