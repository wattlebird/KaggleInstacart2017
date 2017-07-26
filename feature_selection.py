import numpy as np
import pandas as pd
from parametertunning import gbdt_cross_validation_data, gbdt_training, mailsend_feature_selection, mailsendfail
from setting import *


def main():
    try:
        # baseline
        print("Baseline round.")
        aucv = []
        temp = [0.0]*5
        for i, (train, valid, valid_label) in enumerate(gbdt_cross_validation_data()):
            print("\tCV fold {0}.".format(i+1))
            print("\tStart training...")
            model = gbdt_training(params, train, valid, verbose=False)
            print("\tTraining finished.")
            temp[i] = model.best_score['valid_1']['auc']
        aucv.append(np.mean(temp))
        print("Baseline AUC: {0}.".format(aucv[-1]))

        print("Feature dropping round.")
        for singlefeature in feature_list:
            print("\tDropping {0}...".format(singlefeature))
            for i, (train, valid, valid_label) in enumerate(gbdt_cross_validation_data()):
                print("\t\tCV fold {0}.".format(i+1))
                print("\t\tStart training...")
                model = gbdt_training(params, train.drop(singlefeature, axis=1), valid.drop(singlefeature, axis=1), verbose=False)
                print("\t\tTraining finished.")
                temp[i] = model.best_score['valid_1']['auc']
            aucv.append(np.mean(temp))
            print("\tBaseline AUC: {0}.".format(aucv[-1]))
        
        mailsend_feature_selection(feature_list, aucv, params)
    except Exception as e:
        try:
            mailsendfail(title, params, e)
        except Exception as e2:
            raise e

if __name__=="__main__":
    main()