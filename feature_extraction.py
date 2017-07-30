import numpy as np
import pandas as pd
from setting import *
import gc

second_order_ratio = lambda x: x[x==1].count()/x[x==0].count()
gc.enable()

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

    candidate_up = product[['product_id', 'user_id', 'order_time']].groupby(by=['product_id', 'user_id'])['order_time'].agg(['min', 'max', 'count'])
    candidate_up['order_times'] = candidate_up['max']-candidate_up['min']
    candidate_up = candidate_up.drop(['max', 'min'], axis=1).rename(columns={'count': 'user_prod_norder'})
    up1 = pd.merge(product[['product_id', 'user_id', 'order_number', 'interval_accu']], candidate_up, \
                left_on=['product_id', 'user_id'], right_index=True)
    up11 = up1.groupby(by=['product_id', 'user_id'])['order_number'].agg(['min', 'max'])
    up11['order_interval'] = up11['max']-up11['min']
    up11.drop(['max', 'min'], axis=1, inplace=True)
    up12 = up1.groupby(by=['product_id', 'user_id'])['interval_accu'].agg(['min', 'max'])
    up12['days_interval'] = up12['max']-up12['min']
    up12.drop(['max', 'min'], axis=1, inplace=True)
    up1 = pd.merge(candidate_up, up11, left_index=True, right_index=True).merge(up12, left_index=True, right_index=True)
    up1['user_prod_order_interval'] = up1.order_interval/up1.order_times
    up1['user_prod_days_interval'] = up1.days_interval/up1.order_times
    up1.drop(['order_times', 'order_interval', 'days_interval'], axis=1, inplace=True)

    u1 = product[['user_id', 'order_id']].groupby(by='user_id')['order_id'].agg({
        'user_norder': pd.Series.nunique
    })
    up2 = product[['product_id', 'user_id', 'order_number']].groupby(by=['product_id', 'user_id'])['order_number'].agg({
        'user_prod_first_order_time': 'min',
        'user_prod_norder': 'count'
    })
    up2 = pd.merge(up2.reset_index(level=0), u1, left_index=True, right_index=True).reset_index()
    up2['user_prod_reorder_rate'] = (up2.user_prod_norder-1)/(up2.user_norder-up2.user_prod_first_order_time)
    up2 = up2.drop(['user_prod_first_order_time', 'user_norder', 'user_prod_norder'], axis=1).fillna(0)
    u2 = up1.reset_index()[['user_id', 'user_prod_norder']].groupby(by='user_id')['user_prod_norder'].agg({'user_avg_prod_norder': 'mean'})
    u3 = up1.reset_index()[['user_id', 'user_prod_order_interval', 'user_prod_days_interval']].groupby(by='user_id').\
    agg('mean')
    u3 = u3.rename(columns={'user_prod_order_interval': 'user_order_interval_avg', 
                    'user_prod_days_interval': 'user_days_interval_avg'})
    u1 = pd.concat([u1, u2, u3], axis=1)
    p3 = up1.reset_index()[['product_id', 'user_prod_days_interval']].groupby(by='product_id')['user_prod_days_interval'].agg({'prod_days_interval_avg': 'mean'})
    p4 = up1.reset_index()[['product_id', 'user_prod_order_interval']].groupby(by='product_id')['user_prod_order_interval'].agg({'prod_order_interval_avg': 'mean'})
    p3 = pd.concat([p3, p4], axis=1)
    up1 = pd.merge(up1.reset_index(level=0), u1, left_index=True, right_index=True).reset_index().set_index('product_id').\
    merge(p3, left_index=True, right_index=True).reset_index()
    up1 = up1.merge(up2, on=['user_id', 'product_id'])

    p5 = product[['product_id', 'order_time']].groupby(by='product_id')['order_time'].agg({"prod_second_order_ratio": second_order_ratio})

    product_feature = p1.merge(p2, how='left', left_index=True, right_index=True).\
    merge(p3, how='left', left_index=True, right_index=True).\
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

    # user_norder, user_nitems
    u0 = product[['user_id', 'order_id']].groupby(by='user_id')['order_id'].agg({'user_nitems': 'count'})
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
    # user_avg_reorder_ratio
    u6 = product[product['order_number']!=1][['user_id', 'order_id', 'reordered']].\
    groupby(by=['user_id', 'order_id'])['reordered'].agg({
        'dnu': 'count',
        'nu': 'sum'
    })
    u6['reorder_portion'] = u6.nu/u6.dnu
    u6 = u6.reset_index().drop(['order_id', 'dnu', 'nu'], axis=1).groupby(by='user_id')['reorder_portion'].agg({
        'user_avg_reorder_ratio': 'mean'
    })

    user_feature = pd.concat([u0, u1], axis=1).merge(u2, left_index=True, right_index=True).\
    merge(u3, how='left', left_index=True, right_index=True).\
    merge(u4, left_index=True, right_index=True).\
    merge(u5, left_index=True, right_index=True).\
    merge(u6, left_index=True, right_index=True).fillna(0)
    del u0, u1, u2, u3, u4, u5, u6

    user_feature['user_nritem_ratio'] = user_feature.user_nritems/user_feature.user_nitems
    user_feature['user_nrdistinctitem_ratio'] = user_feature.user_nrdistinctitems/user_feature.user_ndistinctitems
    user_feature['user_nitem_per_order'] = user_feature.user_nitems/user_feature.user_norder
    user_feature['user_nritem_per_order'] = user_feature.user_nritems/(user_feature.user_norder-1)
    user_feature['user_nritem_per_order_ratio'] = user_feature.user_nritem_per_order/user_feature.user_nitem_per_order

    up3 = product[['user_id', 'product_id', 'reordered']].groupby(by=['product_id', 'user_id'])['reordered'].agg({
        'user_prod_reordered': 'max'
    })
    ut = product[['user_id', 'order_number']].groupby(by='user_id').agg({'order_number': 'max'})
    up4 = product[['user_id', 'product_id', 'order_number']][product.order_time==0].rename(columns={'order_number': 'first_order_number'})
    up4 = pd.merge(up4, ut, left_on='user_id', right_index=True)
    up4['user_prod_recentlydiscovered'] = pd.Series(up4.first_order_number==up4.order_number)
    up4.drop(['first_order_number', 'order_number'], axis=1, inplace=True)
    user_product_feature = up3.reset_index().merge(up1, how='left', on=['product_id', 'user_id']).\
    merge(up4.astype(np.int), on=['product_id', 'user_id'])
    user_product_feature.user_prod_days_interval = user_product_feature.user_prod_days_interval.fillna(366)
    user_product_feature.user_prod_order_interval = user_product_feature.user_prod_order_interval.fillna(100)
    user_product_feature['user_prod_order_interval_user_ratio'] = user_product_feature.user_prod_order_interval / user_product_feature.user_order_interval_avg
    user_product_feature['user_prod_order_interval_prod_ratio'] = user_product_feature.user_prod_order_interval / user_product_feature.prod_order_interval_avg
    user_product_feature['user_prod_days_interval_user_ratio'] = user_product_feature.user_prod_days_interval / user_product_feature.user_order_interval_avg
    user_product_feature['user_prod_days_interval_prod_ratio'] = user_product_feature.user_prod_days_interval / user_product_feature.prod_order_interval_avg
    user_product_feature['user_prod_norder_user_ratio'] = user_product_feature.user_prod_norder / user_product_feature.user_avg_prod_norder
    user_product_feature['user_prod_order_interval_user_ratio'] = user_product_feature.user_prod_order_interval_user_ratio.fillna(80)
    user_product_feature['user_prod_order_interval_prod_ratio'] = user_product_feature.user_prod_order_interval_prod_ratio.fillna(80)
    user_product_feature['user_prod_days_interval_user_ratio'] = user_product_feature.user_prod_days_interval_user_ratio.fillna(366)
    user_product_feature['user_prod_days_interval_prod_ratio'] = user_product_feature.user_prod_days_interval_prod_ratio.fillna(366)
    user_product_feature.drop(['user_norder', 'user_avg_prod_norder', 'user_order_interval_avg', 'user_days_interval_avg', \
          'prod_days_interval_avg', 'prod_order_interval_avg'], axis=1, inplace=True)
    del up1, up2, up3, up4

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

    train = train.merge(user_feature.reset_index(), on=['user_id'])
    train = train.merge(orders[['order_id', 'user_id', 'days_since_prior_order']][orders.eval_set=='train'], on=['order_id', 'user_id'])
    train.rename(columns={'days_since_prior_order': 'user_lastorder_interval'}, inplace=True)
    train['user_lastorder_interval_ratio'] = train.user_lastorder_interval / train.user_interval
    train.user_lastorder_interval_ratio = train.user_lastorder_interval_ratio.replace(np.inf, np.nan).fillna(1600)

    train = train.merge(orders[['order_id', 'user_id', 'order_number', 'accu_interval']][orders.eval_set=='train'], on=['order_id', 'user_id'])
    train = train.merge(upsp, on=['user_id', 'product_id'])
    train['user_prod_lastorder_interval'] = train.order_number-train.last_order_number
    train['user_prod_lastdays_interval'] = train.accu_interval-train.interval_accu
    train.drop(['order_number', 'accu_interval', 'last_order_number', 'interval_accu'], axis=1, inplace=True)
    train = train.merge(product_feature, left_on='product_id', right_index=True).merge(user_product_feature, on=['user_id', 'product_id'])
    train['user_prod_lastorder_interval_rate'] = train.user_prod_lastorder_interval / train.user_prod_order_interval
    train['user_prod_lastdays_interval_rate']  = train.user_prod_lastdays_interval / train.user_prod_days_interval
    train.user_prod_lastdays_interval_rate = train.user_prod_lastdays_interval_rate.replace(np.inf, np.nan).fillna(1600)

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

    test = test.merge(orders[['order_id', 'user_id', 'order_number', 'accu_interval']][orders.eval_set=='test'], on=['order_id', 'user_id'])
    test = test.merge(upsp, on=['user_id', 'product_id'])
    test['user_prod_lastorder_interval'] = test.order_number-test.last_order_number
    test['user_prod_lastdays_interval'] = test.accu_interval-test.interval_accu
    test.drop(['order_number', 'accu_interval', 'last_order_number', 'interval_accu'], axis=1, inplace=True)
    test = test.merge(product_feature, left_on='product_id', right_index=True).merge(user_product_feature, on=['user_id', 'product_id'])
    test['user_prod_lastorder_interval_rate'] = test.user_prod_lastorder_interval / test.user_prod_order_interval
    test['user_prod_lastdays_interval_rate']  = test.user_prod_lastdays_interval / test.user_prod_days_interval
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

    hour_cnt = product[['order_hour_of_day', 'order_id']].groupby(by='order_hour_of_day')['order_id'].\
    agg({'hour_cnt': 'count'})
    prod_cnt = product[['product_id', 'order_id']].groupby(by='product_id')['order_id'].agg({'prod_cnt':'count'})
    week_cnt = product[['order_dow', 'order_id']].groupby(by='order_dow')['order_id'].agg({'week_cnt':'count'})
    prod_hour_cnt = product[['product_id', 'order_hour_of_day', 'order_id']].groupby(by=['product_id', 'order_hour_of_day'])['order_id'].\
    agg({'prod_hour_cnt': 'count'})
    prod_week_cnt = product[['product_id', 'order_dow', 'order_id']].groupby(by=['product_id', 'order_dow'])['order_id'].\
    agg({'prod_week_cnt': 'count'})

    prod_rcnt = product[product.reordered==1][['product_id', 'order_id']].groupby(by='product_id')['order_id'].\
    agg({'prod_rcnt':'count'})
    prod_hour_rcnt = product[product.reordered==1][['product_id', 'order_hour_of_day', 'order_id']].\
    groupby(by=['product_id', 'order_hour_of_day'])['order_id'].agg({'prod_hour_rcnt': 'count'})
    prod_week_rcnt = product[product.reordered==1][['product_id', 'order_dow', 'order_id']].\
    groupby(by=['product_id', 'order_dow'])['order_id'].agg({'prod_week_rcnt': 'count'})

    user_rcnt = product[product.reordered==1][['user_id', 'order_id']].groupby(by='user_id')['order_id'].\
    agg({'user_rcnt':'count'})
    user_hour_rcnt = product[product.reordered==1][['user_id', 'order_hour_of_day', 'order_id']].\
    groupby(by=['user_id', 'order_hour_of_day'])['order_id'].agg({'user_hour_rcnt': 'count'})
    user_week_rcnt = product[product.reordered==1][['user_id', 'order_dow', 'order_id']].\
    groupby(by=['user_id', 'order_dow'])['order_id'].agg({'user_week_rcnt': 'count'})

    prod_user_rcnt = product[product.reordered==1][['product_id', 'user_id', 'order_id']].groupby(by=['product_id', 'user_id'])['order_id'].\
    agg({'prod_user_rcnt':'count'})
    prod_user_hour_rcnt = product[product.reordered==1][['product_id', 'user_id', 'order_hour_of_day', 'order_id']].\
    groupby(by=['product_id', 'user_id', 'order_hour_of_day'])['order_id'].agg({'prod_user_hour_rcnt': 'count'})
    prod_user_week_rcnt = product[product.reordered==1][['product_id', 'user_id', 'order_dow', 'order_id']].\
    groupby(by=['product_id', 'user_id', 'order_dow'])['order_id'].agg({'prod_user_week_rcnt': 'count'})

    train = pd.read_csv(data+"train.tsv", sep='\t', dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'product_id': np.uint16,
        'label': np.int8
    }, usecols=['order_id', 'user_id', 'product_id'], engine='c')

    train = train.merge(orders[['order_id', 'order_dow', 'order_hour_of_day']][orders.eval_set=='train'], on='order_id').\
    merge(hour_cnt, how='left', left_on='order_hour_of_day', right_index=True).\
    merge(week_cnt, how='left', left_on='order_dow', right_index=True).\
    merge(prod_cnt, how='left', left_on='product_id', right_index=True).fillna(0).\
    merge(prod_hour_cnt, how='left', left_on=['product_id', 'order_hour_of_day'], right_index=True).fillna(0).\
    merge(prod_week_cnt, how='left', left_on=['product_id', 'order_dow'], right_index=True).fillna(0).\
    merge(prod_rcnt, how='left', left_on='product_id', right_index=True).fillna(0).\
    merge(prod_hour_rcnt, how='left', left_on=['product_id', 'order_hour_of_day'], right_index=True).fillna(0).\
    merge(prod_week_rcnt, how='left', left_on=['product_id', 'order_dow'], right_index=True).fillna(0).\
    merge(user_rcnt, how='left', left_on='user_id', right_index=True).fillna(0).\
    merge(user_hour_rcnt, how='left', left_on=['user_id', 'order_hour_of_day'], right_index=True).fillna(0).\
    merge(user_week_rcnt, how='left', left_on=['user_id', 'order_dow'], right_index=True).fillna(0).\
    merge(prod_user_rcnt, how='left', left_on=['product_id', 'user_id'], right_index=True).fillna(0).\
    merge(prod_user_hour_rcnt, how='left', left_on=['product_id', 'user_id', 'order_hour_of_day'], right_index=True).fillna(0).\
    merge(prod_user_week_rcnt, how='left', left_on=['product_id', 'user_id', 'order_dow'], right_index=True).fillna(0)
    train['prod_hour_prob'] = train.prod_hour_cnt / train.hour_cnt
    train['prod_week_prob'] = train.prod_week_cnt / train.week_cnt
    train['week_prod_prob'] = (train.prod_week_cnt+1) / (train.prod_cnt+24)
    train['hour_prod_prob'] = (train.prod_hour_cnt+1) / (train.prod_cnt+7)
    train['hour_user_reorder_prob'] = (train.user_hour_rcnt+1) / (train.user_rcnt+24)
    train['week_user_reorder_prob'] = (train.user_week_rcnt+1) / (train.user_rcnt+7)
    train['hour_prod_reorder_prob'] = (train.prod_hour_rcnt+1) / (train.prod_rcnt+24)
    train['week_prod_reorder_prob'] = (train.prod_week_rcnt+1) / (train.prod_rcnt+7)
    train['hour_prod_user_reorder_prob'] = (train.prod_user_hour_rcnt+1) / (train.prod_user_rcnt+24)
    train['week_prod_user_reorder_prob'] = (train.prod_user_week_rcnt+1) / (train.prod_user_rcnt+7)
    train.drop(['hour_cnt', 'week_cnt', 'prod_cnt', 'prod_hour_cnt', 'prod_week_cnt', 'prod_rcnt', 'prod_hour_rcnt', 
           'prod_week_rcnt', 'user_rcnt', 'user_hour_rcnt', 'user_week_rcnt', 'prod_user_rcnt', 'prod_user_hour_rcnt',
           'prod_user_week_rcnt'], axis=1, inplace=True)

    test = pd.read_csv(data+"test.tsv", sep='\t', dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'product_id': np.uint16
    })
    test = test.merge(orders[['order_id', 'order_dow', 'order_hour_of_day']][orders.eval_set=='test'], on='order_id').\
    merge(hour_cnt, how='left', left_on='order_hour_of_day', right_index=True).\
    merge(week_cnt, how='left', left_on='order_dow', right_index=True).\
    merge(prod_cnt, how='left', left_on='product_id', right_index=True).fillna(0).\
    merge(prod_hour_cnt, how='left', left_on=['product_id', 'order_hour_of_day'], right_index=True).fillna(0).\
    merge(prod_week_cnt, how='left', left_on=['product_id', 'order_dow'], right_index=True).fillna(0).\
    merge(prod_rcnt, how='left', left_on='product_id', right_index=True).fillna(0).\
    merge(prod_hour_rcnt, how='left', left_on=['product_id', 'order_hour_of_day'], right_index=True).fillna(0).\
    merge(prod_week_rcnt, how='left', left_on=['product_id', 'order_dow'], right_index=True).fillna(0).\
    merge(user_rcnt, how='left', left_on='user_id', right_index=True).fillna(0).\
    merge(user_hour_rcnt, how='left', left_on=['user_id', 'order_hour_of_day'], right_index=True).fillna(0).\
    merge(user_week_rcnt, how='left', left_on=['user_id', 'order_dow'], right_index=True).fillna(0).\
    merge(prod_user_rcnt, how='left', left_on=['product_id', 'user_id'], right_index=True).fillna(0).\
    merge(prod_user_hour_rcnt, how='left', left_on=['product_id', 'user_id', 'order_hour_of_day'], right_index=True).fillna(0).\
    merge(prod_user_week_rcnt, how='left', left_on=['product_id', 'user_id', 'order_dow'], right_index=True).fillna(0)
    test['prod_hour_prob'] = test.prod_hour_cnt / test.hour_cnt
    test['prod_week_prob'] = test.prod_week_cnt / test.week_cnt
    test['week_prod_prob'] = (test.prod_week_cnt+1) / (test.prod_cnt+24)
    test['hour_prod_prob'] = (test.prod_hour_cnt+1) / (test.prod_cnt+7)
    test['hour_user_reorder_prob'] = (test.user_hour_rcnt+1) / (test.user_rcnt+24)
    test['week_user_reorder_prob'] = (test.user_week_rcnt+1) / (test.user_rcnt+7)
    test['hour_prod_reorder_prob'] = (test.prod_hour_rcnt+1) / (test.prod_rcnt+24)
    test['week_prod_reorder_prob'] = (test.prod_week_rcnt+1) / (test.prod_rcnt+7)
    test['hour_prod_user_reorder_prob'] = (test.prod_user_hour_rcnt+1) / (test.prod_user_rcnt+24)
    test['week_prod_user_reorder_prob'] = (test.prod_user_week_rcnt+1) / (test.prod_user_rcnt+7)
    test.drop(['hour_cnt', 'week_cnt', 'prod_cnt', 'prod_hour_cnt', 'prod_week_cnt', 'prod_rcnt', 'prod_hour_rcnt', 
           'prod_week_rcnt', 'user_rcnt', 'user_hour_rcnt', 'user_week_rcnt', 'prod_user_rcnt', 'prod_user_hour_rcnt',
           'prod_user_week_rcnt'], axis=1, inplace=True)

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
    product = product.merge(product_detail, on='product_id')

    user_aisle_rcnt = product[product.reordered==1][['user_id', 'aisle_id', 'order_id']].groupby(by=['user_id', 'aisle_id'])['order_id'].\
    agg({'user_aisle_rcnt': 'count'})
    user_dep_rcnt = product[product.reordered==1][['user_id', 'department_id', 'order_id']].\
    groupby(by=['user_id', 'department_id'])['order_id'].agg({'user_dep_rcnt': 'count'})
    user_rcnt = product[product.reordered==1][['user_id', 'order_id']].groupby(by='user_id')['order_id'].agg({'user_rcnt': 'count'})

    prod_aisle_rcnt = product[product.reordered==1][['product_id', 'aisle_id', 'order_id']].groupby(by=['product_id', 'aisle_id'])['order_id'].\
    agg({'prod_aisle_rcnt': 'count'})
    prod_dep_rcnt = product[product.reordered==1][['product_id', 'department_id', 'order_id']].\
    groupby(by=['product_id', 'department_id'])['order_id'].agg({'prod_dep_rcnt': 'count'})
    aisle_cnt = product[product.reordered==1][['aisle_id', 'order_id']].groupby(by='aisle_id')['order_id'].\
    agg({'aisle_cnt': 'count'})
    dep_cnt = product[product.reordered==1][['department_id', 'order_id']].groupby(by='department_id')['order_id'].\
    agg({'dep_cnt': 'count'})

    train = pd.read_csv(data+"train.tsv", sep='\t', dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'product_id': np.uint16,
        'label': np.int8
    }, usecols=['order_id', 'user_id', 'product_id'], engine='c')

    train = train.merge(product_detail, on='product_id').\
    merge(user_aisle_rcnt, how='left', left_on=['user_id', 'aisle_id'], right_index=True).fillna(0).\
    merge(user_dep_rcnt, how='left', left_on=['user_id', 'department_id'], right_index=True).fillna(0).\
    merge(user_rcnt, how='left', left_on='user_id', right_index=True).fillna(0).\
    merge(prod_aisle_rcnt, how='left', left_on=['product_id', 'aisle_id'], right_index=True).fillna(0).\
    merge(prod_dep_rcnt, how='left', left_on=['product_id', 'department_id'], right_index=True).fillna(0).\
    merge(aisle_cnt, how='left', left_on='aisle_id', right_index=True).fillna(0).\
    merge(dep_cnt, how='left', left_on='department_id', right_index=True).fillna(0)
    train['aisle_user_reorder_prob'] = (train.user_aisle_rcnt+1) / (train.user_rcnt+134)
    train['dep_user_reorder_prob'] = (train.user_dep_rcnt+1) / (train.user_rcnt+21)
    train['prod_aisle_reorder_prob'] = train.prod_aisle_rcnt / train.aisle_cnt
    train['prod_dep_reorder_prob'] = train.prod_dep_rcnt / train.dep_cnt
    train.drop(['user_aisle_rcnt', 'user_dep_rcnt', 'user_rcnt', 'prod_aisle_rcnt', 'prod_dep_rcnt', 'aisle_cnt', 'dep_cnt'], axis=1, inplace=True)

    test = pd.read_csv(data+"test.tsv", sep='\t', dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'product_id': np.uint16
    })
    test = test.merge(product_detail, on='product_id').\
    merge(user_aisle_rcnt, how='left', left_on=['user_id', 'aisle_id'], right_index=True).fillna(0).\
    merge(user_dep_rcnt, how='left', left_on=['user_id', 'department_id'], right_index=True).fillna(0).\
    merge(user_rcnt, how='left', left_on='user_id', right_index=True).fillna(0).\
    merge(prod_aisle_rcnt, how='left', left_on=['product_id', 'aisle_id'], right_index=True).fillna(0).\
    merge(prod_dep_rcnt, how='left', left_on=['product_id', 'department_id'], right_index=True).fillna(0).\
    merge(aisle_cnt, how='left', left_on='aisle_id', right_index=True).fillna(0).\
    merge(dep_cnt, how='left', left_on='department_id', right_index=True).fillna(0)
    test['aisle_user_reorder_prob'] = (test.user_aisle_rcnt+1) / (test.user_rcnt+134)
    test['dep_user_reorder_prob'] = (test.user_dep_rcnt+1) / (test.user_rcnt+21)
    test['prod_aisle_reorder_prob'] = test.prod_aisle_rcnt / test.aisle_cnt
    test['prod_dep_reorder_prob'] = test.prod_dep_rcnt / test.dep_cnt
    test.drop(['user_aisle_rcnt', 'user_dep_rcnt', 'user_rcnt', 'prod_aisle_rcnt', 'prod_dep_rcnt', 'aisle_cnt', 'dep_cnt'], axis=1, inplace=True)

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

    #train['aisle_id'] = train.aisle_id.astype('category')
    #train['department_id'] = train.department_id.astype('category')
    #train['order_dow'] = train.order_dow.astype('category')
    #train['order_hour_of_day'] = train.order_hour_of_day.astype('category')
    #train['user_prod_reordered'] = train.user_prod_reordered.astype('category')
    #train['user_prod_recentlydiscovered'] = train.user_prod_recentlydiscovered.astype('category')

    #test['aisle_id'] = test.aisle_id.astype('category')
    #test['department_id'] = test.department_id.astype('category')
    #test['order_dow'] = test.order_dow.astype('category')
    #test['order_hour_of_day'] = test.order_hour_of_day.astype('category')
    #test['user_prod_reordered'] = test.user_prod_reordered.astype('category')
    #test['user_prod_recentlydiscovered'] = test.user_prod_recentlydiscovered.astype('category')

    train.to_hdf(data+"dataset.hdf", "train")
    test.to_hdf(data+"dataset.hdf", "test")