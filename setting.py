import os
from azure.storage.file import FileService, ContentSettings


params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'auc'},
    'num_leaves': 192,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5,
    'verbose': 0
}
data = "/mnt/d/Data/Instacart/"
feature_name = "num_leaves"
title = "GDBT num_leaves tunning experiment"
candidate_param = [96]
file_service = FileService(account_name=os.environ['AZURE_STORAGE_IKELY_ACCOUNT'], 
                                      account_key=os.environ['AZURE_STORAGE_IKELY_KEY'])

def uploadfile(src, dest):
    file_service.create_file_from_path('kaggle',
                                       None, # We want to create this blob in the root directory, so we specify None for the directory_name
                                       dest,
                                       src,
                                       content_settings=ContentSettings(content_type='text/plain'))

__all__ = ['params', 'data', 'feature_name', 'title', 'candidate_param', 'uploadfile']