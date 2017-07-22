import numpy as np
import pandas as pd

data = "/mnt/d/Data/Instacart/"
hdffile = "/mnt/d/Data/Instacart/dataset.hdf"

second_order_ratio = lambda x: x[x==1].count()/x[x==0].count()
avginterval = lambda x: np.inf if x.shape[0]==1 else (x.max()-x.min())/(x.shape[0]-1)

def generate_user_features():
    # data pareparation
    priors = pd.read_csv(data + 'order_products__prior.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8}, usecols = ['order_id', 'product_id', 'reordered'], engine='c')

    orders = pd.read_csv(data + 'orders.csv', dtype={
            'order_id': np.int32,
            'user_id': np.int32,
            'eval_set': 'category',
            'order_number': np.int16,
            'order_dow': np.int8,
            'order_hour_of_day': np.int8,
            'days_since_prior_order': np.float32}, usecols = ['order_id', 'user_id', 'eval_set', 'order_number', 'days_since_prior_order'], engine='c')

    product = pd.merge(priors, orders, on='order_id').sort_values(by=['user_id', 'order_number', 'product_id']).reset_index(drop=True)
    product['order_time']=product.groupby(by=['product_id', 'user_id']).cumcount()
    product.drop('eval_set', axis=1, inplace=True)

    # user_norder, user_nitems
    u1 = product[['user_id', 'order_id']].groupby(by='user_id')['order_id'].agg({'user_norder': pd.Series.nunique, 'user_nitems': 'count'})
    # user_ndistinctitems
    u2 = product[['user_id', 'product_id']].groupby(by='user_id')['product_id'].agg({
        'user_ndistinctitems': pd.Series.nunique
    })
    # user_nritems, user_nrdistinctitems
    u3 = product[['user_id', 'product_id']][product.reordered==1].groupby(by='user_id')['product_id'].agg({
        'user_nritems': 'count',
        'user_nrdistinctitems': pd.Series.nunique
    })
    # user_interval
    u4 = product[['user_id', 'order_id', 'days_since_prior_order']][~product.days_since_prior_order.isnull()].drop_duplicates().\
    drop(['order_id'], axis=1).groupby(by='user_id')['days_since_prior_order'].agg({
        'user_interval': 'mean'
    })
    # user_second_order_rate
    u5 = product[['user_id', 'order_time']].groupby(by='user_id')['order_time'].agg({"user_second_order_rate": second_order_ratio})

    user_feature = u1.merge(u2, left_index=True, right_index=True).\
    merge(u3, how='left', left_index=True, right_index=True).\
    merge(u4, left_index=True, right_index=True).\
    merge(u5, left_index=True, right_index=True).fillna(0)
    del u1, u2, u3, u4, u5

    user_feature['user_nritem_ratio'] = user_feature.user_nritems/user_feature.user_nitems
    user_feature['user_nrdistinctitem_ratio'] = user_feature.user_nrdistinctitems/user_feature.user_ndistinctitems
    user_feature['user_nitem_per_order'] = user_feature.user_nitems/user_feature.user_norder
    user_feature['user_nritem_per_order'] = user_feature.user_nritems/(user_feature.user_norder-1)
    user_feature['user_nritem_per_order_ratio'] = user_feature.user_nritem_per_order/user_feature.user_nitem_per_order

    user_feature[['user_interval', 'user_norder']].to_hdf("/mnt/d/Data/Instacart/dataset.hdf", "user_feature")

    train = pd.read_csv(data+"train.tsv", sep='\t', dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'product_id': np.uint16,
        'label': np.int8
    }, usecols=['order_id', 'user_id', 'product_id'], engine='c')
    train = train.merge(user_feature.reset_index(), on=['user_id'])
    train = train.merge(orders[['order_id', 'user_id', 'days_since_prior_order']][orders.eval_set=='train'], on=['order_id', 'user_id'])
    train.rename(columns={'days_since_prior_order': 'user_lastorder_interval'}, inplace=True)
    train['user_lastorder_interval_ratio'] = train.user_lastorder_interval / train.user_interval
    train.user_lastorder_interval_ratio = train.user_lastorder_interval_ratio.replace(np.inf, np.nan).fillna(1600)

    test = pd.read_csv(data+"test.tsv", sep='\t', dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'product_id': np.uint16
    })
    test = test.merge(user_feature.reset_index(), on=['user_id'])
    test = test.merge(orders[['order_id', 'user_id', 'days_since_prior_order']][orders.eval_set=='test'], on=['order_id', 'user_id'])
    test.rename(columns={'days_since_prior_order': 'user_lastorder_interval'}, inplace=True)
    test['user_lastorder_interval_ratio'] = test.user_lastorder_interval / test.user_interval
    test.user_lastorder_interval_ratio = test.user_lastorder_interval_ratio.replace(np.inf, np.nan).fillna(1600)

    return train, test


def generate_product_user_feature():
    priors = pd.read_csv(data + 'order_products__prior.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8}, usecols = ['order_id', 'product_id', 'reordered'], engine='c')

    orders = pd.read_csv(data + 'orders.csv', dtype={
            'order_id': np.int32,
            'user_id': np.int32,
            'eval_set': 'category',
            'order_number': np.int16,
            'order_dow': np.int8,
            'order_hour_of_day': np.int8,
            'days_since_prior_order': np.float32}, usecols = ['order_id', 'user_id', 'eval_set', 'order_number', 'days_since_prior_order'], engine='c')
    orders['interval_accu'] = orders.groupby(by='user_id')['days_since_prior_order'].cumsum().fillna(0)

    product = pd.merge(priors, orders, on='order_id').sort_values(by=['user_id', 'order_number', 'product_id']).reset_index(drop=True)
    product['order_time']=product.groupby(by=['product_id', 'user_id']).cumcount()

    p1 = product[['product_id', 'user_id', 'order_id']].groupby(by='product_id').agg({'user_id': pd.Series.nunique, 'order_id': pd.Series.nunique}).\
    rename(columns={'user_id': 'prod_nuser', 'order_id': 'prod_norder'})
    p2 = product[['product_id', 'user_id', 'order_id']][product.reordered==1].groupby(by='product_id').agg({'user_id': pd.Series.nunique, 'order_id': pd.Series.nunique}).\
    rename(columns={'user_id': 'prod_nruser', 'order_id': 'prod_nrorder'})
    up1 = product[['product_id', 'user_id', 'interval_accu']].sort_values(by=['user_id', 'product_id', 'interval_accu'])
    up1 = up1.groupby(by=['user_id', 'product_id'])['interval_accu'].agg({'user_prod_days_interval': avginterval})
    up1 = up1[up1.user_prod_days_interval!=np.inf].reset_index()
    p3 = up1[['product_id', 'user_prod_days_interval']].groupby(by='product_id')['user_prod_days_interval'].agg({'prod_days_interval_avg': 'mean'})
    up2 = product[['product_id', 'user_id', 'order_number']].sort_values(by=['user_id', 'product_id', 'order_number'])
    up2 = up2.groupby(by=['product_id', 'user_id'])['order_number'].agg({'user_prod_order_interval': avginterval})
    up2 = up2[up2.user_prod_order_interval!=np.inf].reset_index()
    p4 = up2[['product_id', 'user_prod_order_interval']].groupby(by='product_id')['user_prod_order_interval'].agg({'prod_order_interval_avg': 'mean'})
    p5 = product[['product_id', 'order_time']].groupby(by='product_id')['order_time'].agg({"prod_second_order_ratio": second_order_ratio})

    product_feature = p1.merge(p2, how='left', left_index=True, right_index=True).\
    merge(p3, how='left', left_index=True, right_index=True).\
    merge(p4, how='left', left_index=True, right_index=True).\
    merge(p5, left_index=True, right_index=True)

    product_feature.prod_nruser = product_feature.prod_nruser.fillna(0)
    product_feature.prod_nrorder = product_feature.prod_nrorder.fillna(0)
    product_feature.prod_days_interval_avg = product_feature.prod_days_interval_avg.fillna(366)
    product_feature.prod_order_interval_avg = product_feature.prod_order_interval_avg.fillna(100)
    product_feature['prod_ruser_ratio'] = product_feature.prod_nruser/product_feature.prod_nuser
    product_feature['prod_rorder_ratio'] = product_feature.prod_nrorder/product_feature.prod_norder
    product_feature['prod_rorder_per_ruser'] = product_feature.prod_nrorder/product_feature.prod_nruser
    product_feature['prod_order_per_user'] = product_feature.prod_norder/product_feature.prod_nuser
    product_feature.prod_rorder_per_ruser = product_feature.prod_rorder_per_ruser.fillna(0)
    del p1, p2, p3, p4, p5

    up3 = product[['user_id', 'product_id', 'reordered']].groupby(by=['product_id', 'user_id'])['reordered'].agg({
        'user_prod_norder': 'count',
        'user_prod_reordered': 'max'
    })
    ut = product[['user_id', 'order_number']].groupby(by='user_id').agg({'order_number': 'max'})
    up4 = product[['user_id', 'product_id', 'order_number']][product.order_time==0].rename(columns={'order_number': 'first_order_number'})
    up4 = pd.merge(up4, ut, left_on='user_id', right_index=True)
    up4['user_prod_recentlydiscovered'] = pd.Series(up4.first_order_number==up4.order_number)
    up4.drop(['first_order_number', 'order_number'], axis=1, inplace=True)
    user_product_feature = up3.reset_index().merge(up1, how='left', on=['product_id', 'user_id']).\
    merge(up2, how='left', on=['product_id', 'user_id']).\
    merge(up4.astype(np.int), on=['product_id', 'user_id'])
    user_product_feature.user_prod_days_interval = user_product_feature.user_prod_days_interval.fillna(366)
    user_product_feature.user_prod_order_interval = user_product_feature.user_prod_order_interval.fillna(100)
    del up1, up2, up3, up4

    user_feature = pd.read_hdf("/mnt/d/Data/Instacart/dataset.hdf", "user_feature")
    user_product_feature = user_product_feature.merge(user_feature, left_on='user_id',
                                                 right_index=True)
    user_product_feature['user_prod_norder_rate'] = user_product_feature.user_prod_norder/user_product_feature.user_norder
    user_product_feature['user_prod_days_interval_rate'] = user_product_feature.user_prod_days_interval/user_product_feature.user_interval
    user_product_feature.drop(['user_interval', 'user_norder'], axis=1, inplace=True)
    del user_feature

    upsp = product[['user_id', 'product_id', 'order_time']].groupby(by=['user_id', 'product_id']).agg({'order_time': 'max'}).reset_index()
    upsp = pd.merge(upsp, product[['user_id', 'product_id', 'order_number', 'interval_accu', 'order_time']], on=['user_id', 'product_id', 'order_time'])
    upsp = upsp.rename(columns={'order_number': 'last_order_number'}).drop('order_time', axis=1)
    orders.rename(columns={'interval_accu': 'accu_interval'}, inplace=True)

    train = pd.read_csv(data+"train.tsv", sep='\t', dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'product_id': np.uint16,
        'label': np.int8
    }, usecols=['order_id', 'user_id', 'product_id'], engine='c')
    train = train.merge(orders[['order_id', 'user_id', 'order_number', 'accu_interval']][orders.eval_set=='train'], on=['order_id', 'user_id'])
    train = train.merge(upsp, on=['user_id', 'product_id'])
    train['user_prod_lastorder_interval'] = train.order_number-train.last_order_number
    train['user_prod_lastdays_interval'] = train.accu_interval-train.interval_accu
    train.drop(['order_number', 'accu_interval', 'last_order_number', 'interval_accu'], axis=1, inplace=True)
    train = train.merge(product_feature, left_on='product_id', right_index=True).merge(user_product_feature, on=['user_id', 'product_id'])
    train['user_prod_lastorder_interval_rate'] = train.user_prod_lastorder_interval / train.user_prod_order_interval
    train['user_prod_lastdays_interval_rate']  = train.user_prod_lastdays_interval / train.user_prod_days_interval
    train.user_prod_days_interval_rate = train.user_prod_days_interval_rate.fillna(1)
    train.user_prod_days_interval_rate = train.user_prod_days_interval_rate.replace(np.inf, np.nan).fillna(1600)
    train.user_prod_lastdays_interval_rate = train.user_prod_lastdays_interval_rate.replace(np.inf, np.nan).fillna(1600)

    test = pd.read_csv(data+"test.tsv", sep='\t', dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'product_id': np.uint16
    })
    test = test.merge(orders[['order_id', 'user_id', 'order_number', 'accu_interval']][orders.eval_set=='test'], on=['order_id', 'user_id'])
    test = test.merge(upsp, on=['user_id', 'product_id'])
    test['user_prod_lastorder_interval'] = test.order_number-test.last_order_number
    test['user_prod_lastdays_interval'] = test.accu_interval-test.interval_accu
    test.drop(['order_number', 'accu_interval', 'last_order_number', 'interval_accu'], axis=1, inplace=True)
    test = test.merge(product_feature, left_on='product_id', right_index=True).merge(user_product_feature, on=['user_id', 'product_id'])
    test['user_prod_lastorder_interval_rate'] = test.user_prod_lastorder_interval / test.user_prod_order_interval
    test['user_prod_lastdays_interval_rate']  = test.user_prod_lastdays_interval / test.user_prod_days_interval
    test.user_prod_days_interval_rate = test.user_prod_days_interval_rate.fillna(1)
    test.user_prod_days_interval_rate = test.user_prod_days_interval_rate.replace(np.inf, np.nan).fillna(1600)
    test.user_prod_lastdays_interval_rate = test.user_prod_lastdays_interval_rate.replace(np.inf, np.nan).fillna(1600)

    return train, test

def generate_time_features():
    priors = pd.read_csv(data + 'order_products__prior.csv', dtype={
                'order_id': np.int32,
                'product_id': np.uint16,
                'add_to_cart_order': np.int16,
                'reordered': np.int8}, usecols = ['order_id', 'product_id', 'reordered'], engine='c')

    orders = pd.read_csv(data + 'orders.csv', dtype={
            'order_id': np.int32,
            'user_id': np.int32,
            'eval_set': 'category',
            'order_number': np.int16,
            'order_dow': np.int8,
            'order_hour_of_day': np.int8,
            'days_since_prior_order': np.float32}, usecols = ['order_id', 'user_id', 'eval_set', 'order_number', 'order_dow', 'order_hour_of_day'], engine='c')

    product = pd.merge(priors, orders, on='order_id').sort_values(by=['user_id', 'order_number', 'product_id']).reset_index(drop=True)
    product.drop('eval_set', axis=1, inplace=True)

    t1 = product[['product_id', 'order_hour_of_day', 'order_id']].groupby(by=['product_id', 'order_hour_of_day']).agg('count').reset_index()
    t1 = t1.rename(columns={'order_id': 'hour_cnt'})
    t11 = t1.groupby(by='product_id')['hour_cnt'].agg({'prod_hour_cnt': 'sum'}).reset_index()
    t12 = t1.groupby(by='order_hour_of_day')['hour_cnt'].agg({'hour_prod_cnt': 'sum'}).reset_index()
    t1 = t1.merge(t11, on='product_id').merge(t12, on='order_hour_of_day')
    t1['prod_hour_prob']=t1.hour_cnt/t1.prod_hour_cnt
    t1['hour_prod_prob']=t1.hour_cnt/t1.hour_prod_cnt
    t1.drop(['hour_cnt', 'prod_hour_cnt', 'hour_prod_cnt'], axis=1, inplace=True)
    del t11, t12

    t2 = product[['product_id', 'order_dow', 'order_id']].groupby(by=['product_id', 'order_dow']).agg('count').reset_index()
    t2 = t2.rename(columns={'order_id': 'week_cnt'})
    t21 = t2.groupby(by='product_id')['week_cnt'].agg({'prod_week_cnt': 'sum'}).reset_index()
    t22 = t2.groupby(by='order_dow')['week_cnt'].agg({'week_prod_cnt': 'sum'}).reset_index()
    t2 = t2.merge(t21, on='product_id').merge(t22, on='order_dow')
    t2['prod_week_prob']=t2.week_cnt/t2.prod_week_cnt
    t2['week_prod_prob']=t2.week_cnt/t2.week_prod_cnt
    t2.drop(['week_cnt', 'prod_week_cnt', 'week_prod_cnt'], axis=1, inplace=True)
    del t22, t21

    t3 = product[['user_id', 'order_hour_of_day', 'order_id']][product.reordered==1].\
    groupby(by=['user_id', 'order_hour_of_day'])['order_id'].\
    agg({'user_hour_count': 'count'}).reset_index()
    t31 = product[['user_id', 'order_hour_of_day', 'order_id']][product.reordered==1].\
    groupby(by=['user_id'])['order_id'].\
    agg({'user_count': 'count'}).reset_index()
    t3 = t3.merge(t31, on='user_id')
    t3['hour_user_reorder_prob'] = t3.user_hour_count/t3.user_count
    t3.drop(['user_hour_count', 'user_count'], axis=1, inplace=True)

    t4 = product[['user_id', 'order_dow', 'order_id']][product.reordered==1].\
    groupby(by=['user_id', 'order_dow'])['order_id'].\
    agg({'user_week_count': 'count'}).reset_index()
    t41 = product[['user_id', 'order_dow', 'order_id']][product.reordered==1].\
    groupby(by=['user_id'])['order_id'].\
    agg({'user_count': 'count'}).reset_index()
    t4 = t4.merge(t41, on='user_id')
    t4['week_user_reorder_prob'] = t4.user_week_count/t4.user_count
    t4.drop(['user_week_count', 'user_count'], axis=1, inplace=True)

    t5 = product[['product_id', 'order_hour_of_day', 'order_id']][product.reordered==1].\
    groupby(by=['product_id', 'order_hour_of_day'])['order_id'].\
    agg({'prod_hour_count': 'count'}).reset_index()
    t51 = product[['product_id', 'order_hour_of_day', 'order_id']][product.reordered==1].\
    groupby(by=['product_id'])['order_id'].\
    agg({'prod_count': 'count'}).reset_index()
    t5 = t5.merge(t51, on='product_id')
    t5['hour_prod_reorder_prob'] = t5.prod_hour_count/t5.prod_count
    t5.drop(['prod_hour_count', 'prod_count'], axis=1, inplace=True)

    t6 = product[['product_id', 'order_dow', 'order_id']][product.reordered==1].\
    groupby(by=['product_id', 'order_dow'])['order_id'].\
    agg({'prod_week_count': 'count'}).reset_index()
    t61 = product[['product_id', 'order_dow', 'order_id']][product.reordered==1].\
    groupby(by=['product_id'])['order_id'].\
    agg({'prod_count': 'count'}).reset_index()
    t6 = t6.merge(t61, on='product_id')
    t6['week_prod_reorder_prob'] = t6.prod_week_count/t6.prod_count
    t6.drop(['prod_week_count', 'prod_count'], axis=1, inplace=True)

    t7 = product[['user_id', 'product_id', 'order_hour_of_day', 'order_id']][product.reordered==1].\
    groupby(by=['user_id', 'product_id', 'order_hour_of_day'])['order_id'].\
    agg({'user_prod_hour_count': 'count'}).reset_index()
    t71 = product[['user_id', 'product_id', 'order_hour_of_day', 'order_id']][product.reordered==1].\
    groupby(by=['user_id', 'product_id'])['order_id'].\
    agg({'user_prod_count': 'count'}).reset_index()
    t7 = t7.merge(t71, on=['user_id', 'product_id'])
    t7['hour_user_prod_reorder_prob'] = t7.user_prod_hour_count/t7.user_prod_count
    t7.drop(['user_prod_hour_count', 'user_prod_count'], axis=1, inplace=True)

    t8 = product[['user_id', 'product_id', 'order_dow', 'order_id']][product.reordered==1].\
    groupby(by=['user_id', 'product_id', 'order_dow'])['order_id'].\
    agg({'user_prod_week_count': 'count'}).reset_index()
    t81 = product[['user_id', 'product_id', 'order_dow', 'order_id']][product.reordered==1].\
    groupby(by=['user_id', 'product_id'])['order_id'].\
    agg({'user_prod_count': 'count'}).reset_index()
    t8 = t8.merge(t81, on=['user_id', 'product_id'])
    t8['week_user_prod_reorder_prob'] = t8.user_prod_week_count/t8.user_prod_count
    t8.drop(['user_prod_week_count', 'user_prod_count'], axis=1, inplace=True)

    train = pd.read_csv(data+"train.tsv", sep='\t', dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'product_id': np.uint16,
        'label': np.int8
    }, usecols=['order_id', 'user_id', 'product_id'], engine='c')
    train = train.merge(orders[['order_id', 'order_dow', 'order_hour_of_day']][orders.eval_set=='train'], on='order_id').\
            merge(t1, how='left', on=['product_id', 'order_hour_of_day']).\
            merge(t2, how='left', on=['product_id', 'order_dow']).\
            merge(t3, how='left', on=['user_id', 'order_hour_of_day']).\
            merge(t4, how='left', on=['user_id', 'order_dow']).\
            merge(t5, how='left', on=['product_id', 'order_hour_of_day']).\
            merge(t6, how='left', on=['product_id', 'order_dow']).fillna(0).\
            merge(t7, how='left', on=['user_id', 'product_id', 'order_hour_of_day']).\
            merge(t8, how='left', on=['user_id', 'product_id', 'order_dow']).fillna(0)

    test = pd.read_csv(data+"test.tsv", sep='\t', dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'product_id': np.uint16
    })
    test = test.merge(orders[['order_id', 'order_dow', 'order_hour_of_day']][orders.eval_set=='test'], on='order_id').merge(t1, how='left', on=['product_id', 'order_hour_of_day']).\
    merge(t2, how='left', on=['product_id', 'order_dow']).\
    merge(t3, how='left', on=['user_id', 'order_hour_of_day']).\
    merge(t4, how='left', on=['user_id', 'order_dow']).\
    merge(t5, how='left', on=['product_id', 'order_hour_of_day']).\
    merge(t6, how='left', on=['product_id', 'order_dow']).fillna(0).\
    merge(t7, how='left', on=['user_id', 'product_id', 'order_hour_of_day']).\
    merge(t8, how='left', on=['user_id', 'product_id', 'order_dow']).fillna(0)

    return train, test

def generate_id_features():
    # asile and department
    product_detail = pd.read_csv(data+"products.csv", dtype = {
        'product_id': np.uint16,
        'product_name': str,
        'aisle_id': np.uint8,
        'department_id': np.uint8
    }, usecols = ['product_id', 'aisle_id', 'department_id'])

    priors = pd.read_csv(data + 'order_products__prior.csv', dtype={
                'order_id': np.int32,
                'product_id': np.uint16,
                'add_to_cart_order': np.int16,
                'reordered': np.int8}, usecols = ['order_id', 'product_id', 'reordered'], engine='c')

    orders = pd.read_csv(data + 'orders.csv', dtype={
            'order_id': np.int32,
            'user_id': np.int32,
            'eval_set': 'category',
            'order_number': np.int16,
            'order_dow': np.int8,
            'order_hour_of_day': np.int8,
            'days_since_prior_order': np.float32}, usecols = ['order_id', 'user_id', 'eval_set', 'order_number'], engine='c')

    product = pd.merge(priors, orders, on='order_id').sort_values(by=['user_id', 'order_number', 'product_id']).reset_index(drop=True)
    product.drop('eval_set', axis=1, inplace=True)

    d1 = pd.merge(product_detail, product[['product_id', 'order_id']][product.reordered==1], on='product_id').groupby(by=['product_id', 'aisle_id', 'department_id']).\
    agg({'order_id': 'count'}).reset_index()
    d11 = d1[['aisle_id', 'order_id']].groupby(by='aisle_id')['order_id'].agg({'aisle_count': 'sum'})
    d12 = d1[['department_id', 'order_id']].groupby(by='department_id')['order_id'].agg({'department_count': 'sum'})
    d1 = d1.merge(d11, left_on='aisle_id', right_index=True).merge(d12, left_on='department_id', right_index=True)
    d1['prod_aisle_reorder_prob'] = d1.order_id/d1.aisle_count
    d1['prod_department_reorder_prob'] = d1.order_id/d1.department_count
    d1.drop(['order_id', 'aisle_count', 'department_count'], axis=1, inplace=True)

    d = pd.merge(product_detail, product[['product_id', 'user_id', 'order_id']][product.reordered==1], on='product_id')

    d2 = d[['user_id', 'order_id']].groupby(by='user_id')['order_id'].agg({'user_rorder_cnt': 'count'}).reset_index()
    d21 = d[['user_id', 'aisle_id', 'order_id']].groupby(by=['user_id', 'aisle_id'])['order_id'].agg({'aisle_count': 'count'}).reset_index()
    d22 = d[['user_id', 'department_id', 'order_id']].groupby(by=['user_id', 'department_id'])['order_id'].agg({'department_count': 'count'}).reset_index()
    d2 = d2.merge(d21, on='user_id').merge(d22, on='user_id')
    d2['aisle_user_reorder_prob'] = d2.aisle_count/d2.user_rorder_cnt
    d2['department_user_reorder_prob'] = d2.department_count/d2.user_rorder_cnt
    d2.drop(['user_rorder_cnt', 'aisle_count', 'department_count'], axis=1, inplace=True)

    train = pd.read_csv(data+"train.tsv", sep='\t', dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'product_id': np.uint16,
        'label': np.int8
    }, usecols=['order_id', 'user_id', 'product_id'], engine='c')
    train = train.merge(product_detail, on='product_id').merge(d1, how='left', on=['product_id', 'aisle_id', 'department_id']).fillna(0).\
    merge(d2, how='left', on=['user_id', 'aisle_id', 'department_id']).fillna(0)

    test = pd.read_csv(data+"test.tsv", sep='\t', dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'product_id': np.uint16
    })
    test = test.merge(product_detail, on='product_id').merge(d1, how='left', on=['product_id', 'aisle_id', 'department_id']).fillna(0).\
    merge(d2, how='left', on=['user_id', 'aisle_id', 'department_id']).fillna(0)

    return train, test

if __name__=="__main__":

    train = pd.read_csv(data+"train.tsv", sep='\t', dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'product_id': np.uint16,
        'label': np.int8
    }, engine='c')
    test = pd.read_csv(data+"test.tsv", sep='\t', dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'product_id': np.uint16
    })

    trainu, testu = generate_user_features()
    train = train.merge(trainu, on=['order_id', 'user_id', 'product_id'])
    test = test.merge(testu, on=['order_id', 'user_id', 'product_id'])
    del trainu, testu
    trainp, testp = generate_product_user_feature()
    train = train.merge(trainp, on=['order_id', 'user_id', 'product_id'])
    test = test.merge(testp, on=['order_id', 'user_id', 'product_id'])
    del trainp, testp
    traint, testt = generate_time_features()
    train = train.merge(traint, on=['order_id', 'user_id', 'product_id'])
    test = test.merge(testt, on=['order_id', 'user_id', 'product_id'])
    del traint, testt
    traini, testi = generate_id_features()
    train = train.merge(traini, on=['order_id', 'user_id', 'product_id'])
    test = test.merge(testi, on=['order_id', 'user_id', 'product_id'])
    del traini, testi

    train['aisle_id'] = train.aisle_id.astype('category')
    train['department_id'] = train.department_id.astype('category')
    train['order_dow'] = train.order_dow.astype('category')
    train['order_hour_of_day'] = train.order_hour_of_day.astype('category')
    train['user_prod_reordered'] = train.user_prod_reordered.astype('category')
    train['user_prod_recentlydiscovered'] = train.user_prod_recentlydiscovered.astype('category')

    test['aisle_id'] = test.aisle_id.astype('category')
    test['department_id'] = test.department_id.astype('category')
    test['order_dow'] = test.order_dow.astype('category')
    test['order_hour_of_day'] = test.order_hour_of_day.astype('category')
    test['user_prod_reordered'] = test.user_prod_reordered.astype('category')
    test['user_prod_recentlydiscovered'] = test.user_prod_recentlydiscovered.astype('category')

    train.to_hdf("/mnt/d/Data/Instacart/dataset.hdf", "train")
    test.to_hdf("/mnt/d/Data/Instacart/dataset.hdf", "test")