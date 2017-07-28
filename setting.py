import os
from azure.storage.file import FileService, ContentSettings
from sklearn.metrics import f1_score


params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'auc'},
    'num_leaves': 224,
    'min_data_in_leaf': 4500,
    'feature_fraction': 0.9,
    'bagging_fraction': 1,
    'verbose': 0
}
data = "/home/azureuser/Data/"
feature_name = "feature_fraction"
title = "GDBT feature_fraction tunning experiment"
candidate_param = [0.6, 0.7, 0.8, 0.9]
file_service = FileService(account_name=os.environ['AZURE_STORAGE_IKELY_ACCOUNT'], 
                                      account_key=os.environ['AZURE_STORAGE_IKELY_KEY'])
feature_list = [#'user_nitems', 'user_norder', 'user_ndistinctitems',
       #'user_nritems', 'user_nrdistinctitems', 'user_interval',
       #'user_second_order_rate', 'user_nritem_ratio',
       #'user_nrdistinctitem_ratio', 'user_nitem_per_order',
       #'user_nritem_per_order', 'user_nritem_per_order_ratio',
       #'user_lastorder_interval', 'user_lastorder_interval_ratio',
       #'user_prod_lastorder_interval', 'user_prod_lastdays_interval',
       'prod_norder', 'prod_nuser', 'prod_nrorder', 'prod_nruser',
       'prod_days_interval_avg', 'prod_order_interval_avg',
       'prod_second_order_ratio', 'prod_ruser_ratio', 'prod_rorder_ratio',
       'prod_rorder_per_ruser', 'prod_order_per_user', 'user_prod_norder',
       'user_prod_reordered', 'user_prod_days_interval',
       'user_prod_order_interval', 'user_prod_recentlydiscovered',
       'user_prod_norder_rate', 'user_prod_days_interval_rate',
       'user_prod_lastorder_interval_rate', 'user_prod_lastdays_interval_rate',
       'order_dow', 'order_hour_of_day', 'prod_hour_prob', 'hour_prod_prob',
       'prod_week_prob', 'week_prod_prob', 'hour_user_reorder_prob',
       'week_user_reorder_prob', 'hour_prod_reorder_prob',
       'week_prod_reorder_prob', 'hour_user_prod_reorder_prob',
       'week_user_prod_reorder_prob', 'aisle_id', 'department_id',
       'prod_aisle_reorder_prob', 'prod_department_reorder_prob',
       'aisle_user_reorder_prob', 'department_user_reorder_prob']

def uploadfile(src, dest):
    file_service.create_file_from_path('kaggle',
                                       None, # We want to create this blob in the root directory, so we specify None for the directory_name
                                       dest,
                                       src,
                                       content_settings=ContentSettings(content_type='text/plain'))

def uf1score(x, y):
    return f1_score(x.label, x.pred_prob>y)

__all__ = ['params', 'data', 'feature_name', 'title', 'candidate_param', 'uploadfile', 'feature_list', 'uf1score']
