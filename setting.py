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
    'lambda_l2':0,
    'verbose': 0
}

rf_params = {
    'task': 'train',
    'boosting_type': 'rf',
    'objective': 'binary',
    'metric': {'auc'},
    'num_leaves': 2048,
    'lambda_l1': 0.03125,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.6,
    'bagging_freq': 1,
    'verbose': 0
}

data = "/home/azureuser/Data/"
feature_name = "lambda_l2"
title = "GBDT lambda_l2 tunning experiment"
candidate_param = [0.03125, 0.25, 1, 4, 16]
file_service = FileService(account_name=os.environ['AZURE_STORAGE_IKELY_ACCOUNT'], 
                                      account_key=os.environ['AZURE_STORAGE_IKELY_KEY'])
feature_list = [
"user_prod_recentlydiscovered",
"hour_prod_user_reorder_prob",
"user_prod_order_interval",
"week_prod_user_reorder_prob",
"user_nritem_ratio",
"prod_norder",
"prod_order_per_user",
"prod_nruser",
"prod_ruser_ratio",
"user_prod_days_interval",
"user_prod_days_interval_prod_ratio",
"prod_week_prob",
"user_nrdistinctitem_ratio",
"prod_hour_prob",
"order_dow",
"prod_dep_reorder_prob",
"user_prod_order_interval_prod_ratio",
"user_prod_order_interval_user_ratio",
"user_prod_norder",
"prod_nrorder",
"department_id",
"user_prod_days_interval_user_ratio",
"order_hour_of_day",
"prod_aisle_reorder_prob",
"prod_nuser",
"hour_prod_reorder_prob",
"prod_rorder_per_ruser",
"week_prod_reorder_prob",
"week_prod_prob",
"user_prod_lastorder_interval_rate",
"user_prod_norder_user_ratio",
"user_nritems",
"prod_rorder_ratio",
"aisle_id",
"hour_prod_prob",
"user_avg_prod_norder",
"user_nitems",
"user_prod_reorder_rate",
"user_nrdistinctitems",
"aisle_user_reorder_prob",
"user_prod_lastorder_interval",
"dep_user_reorder_prob"
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
