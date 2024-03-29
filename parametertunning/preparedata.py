import numpy as np
import pandas as pd
from setting import *

def gbdt_cross_validation_data():
    f = pd.read_hdf(data+"dataset.hdf", "train")
    f['aisle_id'] = f.aisle_id.astype('category')
    f['department_id'] = f.department_id.astype('category')
    f['order_dow'] = f.order_dow.astype('category')
    f['order_hour_of_day'] = f.order_hour_of_day.astype('category')
    f['user_prod_reordered'] = f.user_prod_reordered.astype('category')
    f['user_prod_recentlydiscovered'] = f.user_prod_recentlydiscovered.astype('category')
    if len(forbid_feature_list)!=0:
        f.drop(forbid_feature_list, 
                axis=1, inplace=True)
    for i in range(5):
        train = f[f['seed']!=i].drop(['order_id', 'user_id', 'product_id', 'seed'], axis=1)
        valid = f[f['seed']==i].drop(['order_id', 'user_id', 'product_id', 'seed'], axis=1)
        valid_label = f[['order_id', 'user_id', 'product_id', 'label']][f['seed']==i]
        yield train, valid, valid_label

def gbdt_cross_validation_data_debug():
    f = pd.read_hdf(data+"dataset.hdf", "train").head(10000)
    f['aisle_id'] = f.aisle_id.astype('category')
    f['department_id'] = f.department_id.astype('category')
    f['order_dow'] = f.order_dow.astype('category')
    f['order_hour_of_day'] = f.order_hour_of_day.astype('category')
    f['user_prod_reordered'] = f.user_prod_reordered.astype('category')
    f['user_prod_recentlydiscovered'] = f.user_prod_recentlydiscovered.astype('category')
    if len(forbid_feature_list)!=0:
        f.drop(forbid_feature_list, 
                axis=1, inplace=True)
    for i in range(5):
        train = f[f['seed']!=i].drop(['order_id', 'user_id', 'product_id', 'seed'], axis=1)
        valid = f[f['seed']==i].drop(['order_id', 'user_id', 'product_id', 'seed'], axis=1)
        valid_label = f[['order_id', 'user_id', 'product_id', 'label']][f['seed']==i]
        yield train, valid, valid_label

def gbdt_get_training_data():
    f = pd.read_hdf(data+"dataset.hdf", "train")
    f['aisle_id'] = f.aisle_id.astype('category')
    f['department_id'] = f.department_id.astype('category')
    f['order_dow'] = f.order_dow.astype('category')
    f['order_hour_of_day'] = f.order_hour_of_day.astype('category')
    f['user_prod_reordered'] = f.user_prod_reordered.astype('category')
    f['user_prod_recentlydiscovered'] = f.user_prod_recentlydiscovered.astype('category')
    if len(forbid_feature_list)!=0:
        f.drop(forbid_feature_list, 
                axis=1, inplace=True)
    return f

def gbdt_get_testing_data():
    f = pd.read_hdf(data+"dataset.hdf", "test")
    f['aisle_id'] = f.aisle_id.astype('category')
    f['department_id'] = f.department_id.astype('category')
    f['order_dow'] = f.order_dow.astype('category')
    f['order_hour_of_day'] = f.order_hour_of_day.astype('category')
    f['user_prod_reordered'] = f.user_prod_reordered.astype('category')
    f['user_prod_recentlydiscovered'] = f.user_prod_recentlydiscovered.astype('category')
    if len(forbid_feature_list)!=0:
        f.drop(forbid_feature_list, 
                axis=1, inplace=True)
    return f