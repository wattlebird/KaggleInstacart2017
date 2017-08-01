import os
from azure.storage.file import FileService, ContentSettings
from sklearn.metrics import f1_score


gbdt_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'auc'},
    'num_leaves': 224,
    'min_data_in_leaf': 4500,
    'feature_fraction': 0.7,
    'bagging_fraction': 1,
    'verbose': 0
}

rf_params = {
    'task': 'train',
    'boosting_type': 'rf',
    'objective': 'binary',
    'metric': {'auc'},
    'num_leaves': 2048,
    'lambda_l1': 4,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.6,
    'bagging_freq': 1,
    'verbose': 0
}

data = "/home/azureuser/Data/"
feature_name = "lambda_l1"
title = "RF lambda_l1 tunning experiment"
candidate_param = [0.03125, 0.25, 1, 4, 16]
file_service = FileService(account_name=os.environ['AZURE_STORAGE_IKELY_ACCOUNT'], 
                                      account_key=os.environ['AZURE_STORAGE_IKELY_KEY'])
feature_list = [
user_interval, 
prod_days_interval_avg, 
user_nritem_per_order, 
user_ndistinctitems, 
user_nitem_per_order, 
user_avg_reorder_ratio, 
user_lastorder_interval, 
week_user_reorder_prob, 
hour_user_reorder_prob, 
user_norder, 
user_nritem_per_order_ratio, 
user_prod_lastdays_interval_rate, 
prod_second_order_ratio, 
prod_order_interval_avg, 
dep_user_reorder_prob, 
user_prod_lastorder_interval, 
aisle_user_reorder_prob, 
user_nrdistinctitems, 
user_prod_reorder_rate, 
user_nitems, 
user_avg_prod_norder, 
hour_prod_prob, 
aisle_id, 
prod_rorder_ratio, 
user_nritems, 
user_prod_norder_user_ratio, 
user_prod_lastorder_interval_rate, 
week_prod_prob, 
week_prod_reorder_prob, 
prod_rorder_per_ruser, 
hour_prod_reorder_prob, 
prod_nuser, 
prod_aisle_reorder_prob, 
order_hour_of_day, 
user_prod_days_interval_user_ratio, 
department_id, 
prod_nrorder, 
user_prod_norder, 
user_prod_order_interval_user_ratio, 
user_prod_order_interval_prod_ratio, 
prod_dep_reorder_prob, 
order_dow, 
prod_hour_prob, 
user_nrdistinctitem_ratio, 
prod_week_prob, 
user_prod_days_interval_prod_ratio, 
user_prod_days_interval, 
prod_ruser_ratio, 
prod_nruser, 
prod_order_per_user, 
prod_norder, 
user_nritem_ratio, 
week_prod_user_reorder_prob, 
user_prod_order_interval, 
hour_prod_user_reorder_prob, 
user_prod_recentlydiscovered
]

forbid_feature_list = [
]

TH = 0.177

def uploadfile(src, dest):
    file_service.create_file_from_path('kaggle',
                                       None, # We want to create this blob in the root directory, so we specify None for the directory_name
                                       dest,
                                       src,
                                       content_settings=ContentSettings(content_type='text/plain'))

def uf1score(x, y):
    return f1_score(x.label, x.pred_prob>y)

__all__ = ['gbdt_params', 'data', 'feature_name', 'title', 'candidate_param', 'uploadfile', 'feature_list', 'uf1score',
'forbid_feature_list', 'rf_params', 'TH']
