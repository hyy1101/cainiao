# coding: utf-8
# filename: model.py  
# author: hyy1101.github.io

import pandas as pd
import numpy as np
import xgboost as xgb
import re
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import os,cPickle,random
config = pd.read_csv('../data/config2.csv',header=None)

train_all = pd.read_csv('/Applications/dev/tmp/train_all.csv',header=None)
train_all.head(5)

# 数据加载以及处理
train_all = pd.read_csv('/Applications/dev/tmp/train_all.csv',header=None)
train_all.columns = ['watch','store_code','item_id','label','pv_ipv_1','pv_uv_1','cart_ipv_1','cart_uv_1','collect_uv_1','num_gmv_1','amt_gmv_1','qty_gmv_1','unum_gmv_1','amt_alipay_1','num_alipay_1','qty_alipay_1','unum_alipay_1','ztc_pv_ipv_1','tbk_pv_ipv_1','ss_pv_ipv_1','jhs_pv_ipv_1','ztc_pv_uv_1','tbk_pv_uv_1','ss_pv_uv_1','jhs_pv_uv_1','num_alipay_njhs_1','amt_alipay_njhs_1','qty_alipay_njhs_1','unum_alipay_njhs_1','pv_ipv_sum_1','pv_uv_sum_1','cart_ipv_sum_1','cart_uv_sum_1','collect_uv_sum_1','num_gmv_sum_1','amt_gmv_sum_1','qty_gmv_sum_1','unum_gmv_sum_1','amt_alipay_sum_1','num_alipay_sum_1','qty_alipay_sum_1','unum_alipay_sum_1','ztc_pv_ipv_sum_1','tbk_pv_ipv_sum_1','ss_pv_ipv_sum_1','jhs_pv_ipv_sum_1','ztc_pv_uv_sum_1','tbk_pv_uv_sum_1','ss_pv_uv_sum_1','jhs_pv_uv_sum_1','num_alipay_njhs_sum_1','amt_alipay_njhs_sum_1','qty_alipay_njhs_sum_1','unum_alipay_njhs_sum_1','pv_ipv_2','pv_uv_2','cart_ipv_2','cart_uv_2','collect_uv_2','num_gmv_2','amt_gmv_2','qty_gmv_2','unum_gmv_2','amt_alipay_2','num_alipay_2','qty_alipay_2','unum_alipay_2','ztc_pv_ipv_2','tbk_pv_ipv_2','ss_pv_ipv_2','jhs_pv_ipv_2','ztc_pv_uv_2','tbk_pv_uv_2','ss_pv_uv_2','jhs_pv_uv_2','num_alipay_njhs_2','amt_alipay_njhs_2','qty_alipay_njhs_2','unum_alipay_njhs_2','pv_ipv_sum_2','pv_uv_sum_2','cart_ipv_sum_2','cart_uv_sum_2','collect_uv_sum_2','num_gmv_sum_2','amt_gmv_sum_2','qty_gmv_sum_2','unum_gmv_sum_2','amt_alipay_sum_2','num_alipay_sum_2','qty_alipay_sum_2','unum_alipay_sum_2','ztc_pv_ipv_sum_2','tbk_pv_ipv_sum_2','ss_pv_ipv_sum_2','jhs_pv_ipv_sum_2','ztc_pv_uv_sum_2','tbk_pv_uv_sum_2','ss_pv_uv_sum_2','jhs_pv_uv_sum_2','num_alipay_njhs_sum_2','amt_alipay_njhs_sum_2','qty_alipay_njhs_sum_2','unum_alipay_njhs_sum_2','pv_ipv_3','pv_uv_3','cart_ipv_3','cart_uv_3','collect_uv_3','num_gmv_3','amt_gmv_3','qty_gmv_3','unum_gmv_3','amt_alipay_3','num_alipay_3','qty_alipay_3','unum_alipay_3','ztc_pv_ipv_3','tbk_pv_ipv_3','ss_pv_ipv_3','jhs_pv_ipv_3','ztc_pv_uv_3','tbk_pv_uv_3','ss_pv_uv_3','jhs_pv_uv_3','num_alipay_njhs_3','amt_alipay_njhs_3','qty_alipay_njhs_3','unum_alipay_njhs_3','pv_ipv_sum_3','pv_uv_sum_3','cart_ipv_sum_3','cart_uv_sum_3','collect_uv_sum_3','num_gmv_sum_3','amt_gmv_sum_3','qty_gmv_sum_3','unum_gmv_sum_3','amt_alipay_sum_3','num_alipay_sum_3','qty_alipay_sum_3','unum_alipay_sum_3','ztc_pv_ipv_sum_3','tbk_pv_ipv_sum_3','ss_pv_ipv_sum_3','jhs_pv_ipv_sum_3','ztc_pv_uv_sum_3','tbk_pv_uv_sum_3','ss_pv_uv_sum_3','jhs_pv_uv_sum_3','num_alipay_njhs_sum_3','amt_alipay_njhs_sum_3','qty_alipay_njhs_sum_3','unum_alipay_njhs_sum_3','pv_ipv_5','pv_uv_5','cart_ipv_5','cart_uv_5','collect_uv_5','num_gmv_5','amt_gmv_5','qty_gmv_5','unum_gmv_5','amt_alipay_5','num_alipay_5','qty_alipay_5','unum_alipay_5','ztc_pv_ipv_5','tbk_pv_ipv_5','ss_pv_ipv_5','jhs_pv_ipv_5','ztc_pv_uv_5','tbk_pv_uv_5','ss_pv_uv_5','jhs_pv_uv_5','num_alipay_njhs_5','amt_alipay_njhs_5','qty_alipay_njhs_5','unum_alipay_njhs_5','pv_ipv_sum_5','pv_uv_sum_5','cart_ipv_sum_5','cart_uv_sum_5','collect_uv_sum_5','num_gmv_sum_5','amt_gmv_sum_5','qty_gmv_sum_5','unum_gmv_sum_5','amt_alipay_sum_5','num_alipay_sum_5','qty_alipay_sum_5','unum_alipay_sum_5','ztc_pv_ipv_sum_5','tbk_pv_ipv_sum_5','ss_pv_ipv_sum_5','jhs_pv_ipv_sum_5','ztc_pv_uv_sum_5','tbk_pv_uv_sum_5','ss_pv_uv_sum_5','jhs_pv_uv_sum_5','num_alipay_njhs_sum_5','amt_alipay_njhs_sum_5','qty_alipay_njhs_sum_5','unum_alipay_njhs_sum_5','pv_ipv_7','pv_uv_7','cart_ipv_7','cart_uv_7','collect_uv_7','num_gmv_7','amt_gmv_7','qty_gmv_7','unum_gmv_7','amt_alipay_7','num_alipay_7','qty_alipay_7','unum_alipay_7','ztc_pv_ipv_7','tbk_pv_ipv_7','ss_pv_ipv_7','jhs_pv_ipv_7','ztc_pv_uv_7','tbk_pv_uv_7','ss_pv_uv_7','jhs_pv_uv_7','num_alipay_njhs_7','amt_alipay_njhs_7','qty_alipay_njhs_7','unum_alipay_njhs_7','pv_ipv_sum_7','pv_uv_sum_7','cart_ipv_sum_7','cart_uv_sum_7','collect_uv_sum_7','num_gmv_sum_7','amt_gmv_sum_7','qty_gmv_sum_7','unum_gmv_sum_7','amt_alipay_sum_7','num_alipay_sum_7','qty_alipay_sum_7','unum_alipay_sum_7','ztc_pv_ipv_sum_7','tbk_pv_ipv_sum_7','ss_pv_ipv_sum_7','jhs_pv_ipv_sum_7','ztc_pv_uv_sum_7','tbk_pv_uv_sum_7','ss_pv_uv_sum_7','jhs_pv_uv_sum_7','num_alipay_njhs_sum_7','amt_alipay_njhs_sum_7','qty_alipay_njhs_sum_7','unum_alipay_njhs_sum_7','pv_ipv_9','pv_uv_9','cart_ipv_9','cart_uv_9','collect_uv_9','num_gmv_9','amt_gmv_9','qty_gmv_9','unum_gmv_9','amt_alipay_9','num_alipay_9','qty_alipay_9','unum_alipay_9','ztc_pv_ipv_9','tbk_pv_ipv_9','ss_pv_ipv_9','jhs_pv_ipv_9','ztc_pv_uv_9','tbk_pv_uv_9','ss_pv_uv_9','jhs_pv_uv_9','num_alipay_njhs_9','amt_alipay_njhs_9','qty_alipay_njhs_9','unum_alipay_njhs_9','pv_ipv_sum_9','pv_uv_sum_9','cart_ipv_sum_9','cart_uv_sum_9','collect_uv_sum_9','num_gmv_sum_9','amt_gmv_sum_9','qty_gmv_sum_9','unum_gmv_sum_9','amt_alipay_sum_9','num_alipay_sum_9','qty_alipay_sum_9','unum_alipay_sum_9','ztc_pv_ipv_sum_9','tbk_pv_ipv_sum_9','ss_pv_ipv_sum_9','jhs_pv_ipv_sum_9','ztc_pv_uv_sum_9','tbk_pv_uv_sum_9','ss_pv_uv_sum_9','jhs_pv_uv_sum_9','num_alipay_njhs_sum_9','amt_alipay_njhs_sum_9','qty_alipay_njhs_sum_9','unum_alipay_njhs_sum_9','pv_ipv_11','pv_uv_11','cart_ipv_11','cart_uv_11','collect_uv_11','num_gmv_11','amt_gmv_11','qty_gmv_11','unum_gmv_11','amt_alipay_11','num_alipay_11','qty_alipay_11','unum_alipay_11','ztc_pv_ipv_11','tbk_pv_ipv_11','ss_pv_ipv_11','jhs_pv_ipv_11','ztc_pv_uv_11','tbk_pv_uv_11','ss_pv_uv_11','jhs_pv_uv_11','num_alipay_njhs_11','amt_alipay_njhs_11','qty_alipay_njhs_11','unum_alipay_njhs_11','pv_ipv_sum_11','pv_uv_sum_11','cart_ipv_sum_11','cart_uv_sum_11','collect_uv_sum_11','num_gmv_sum_11','amt_gmv_sum_11','qty_gmv_sum_11','unum_gmv_sum_11','amt_alipay_sum_11','num_alipay_sum_11','qty_alipay_sum_11','unum_alipay_sum_11','ztc_pv_ipv_sum_11','tbk_pv_ipv_sum_11','ss_pv_ipv_sum_11','jhs_pv_ipv_sum_11','ztc_pv_uv_sum_11','tbk_pv_uv_sum_11','ss_pv_uv_sum_11','jhs_pv_uv_sum_11','num_alipay_njhs_sum_11','amt_alipay_njhs_sum_11','qty_alipay_njhs_sum_11','unum_alipay_njhs_sum_11','pv_ipv_14','pv_uv_14','cart_ipv_14','cart_uv_14','collect_uv_14','num_gmv_14','amt_gmv_14','qty_gmv_14','unum_gmv_14','amt_alipay_14','num_alipay_14','qty_alipay_14','unum_alipay_14','ztc_pv_ipv_14','tbk_pv_ipv_14','ss_pv_ipv_14','jhs_pv_ipv_14','ztc_pv_uv_14','tbk_pv_uv_14','ss_pv_uv_14','jhs_pv_uv_14','num_alipay_njhs_14','amt_alipay_njhs_14','qty_alipay_njhs_14','unum_alipay_njhs_14','pv_ipv_sum_14','pv_uv_sum_14','cart_ipv_sum_14','cart_uv_sum_14','collect_uv_sum_14','num_gmv_sum_14','amt_gmv_sum_14','qty_gmv_sum_14','unum_gmv_sum_14','amt_alipay_sum_14','num_alipay_sum_14','qty_alipay_sum_14','unum_alipay_sum_14','ztc_pv_ipv_sum_14','tbk_pv_ipv_sum_14','ss_pv_ipv_sum_14','jhs_pv_ipv_sum_14','ztc_pv_uv_sum_14','tbk_pv_uv_sum_14','ss_pv_uv_sum_14','jhs_pv_uv_sum_14','num_alipay_njhs_sum_14','amt_alipay_njhs_sum_14','qty_alipay_njhs_sum_14','unum_alipay_njhs_sum_14','max_qty_alipay_njhs','min_qty_alipay_njhs','std_qty_alipay_njhs','cate_id_qty_alipay_njhs_sum','cate_id_qty_alipay_njhs_avg','cate_id_qty_alipay_njhs_std','cate_level_id_qty_alipay_njhs_sum','cate_level_id_qty_alipay_njhs_avg','cate_level_id_qty_alipay_njhs_std','brand_id_qty_alipay_njhs_sum','brand_id_qty_alipay_njhs_avg','brand_id_qty_alipay_njhs_std','supplier_id_qty_alipay_njhs_sum','supplier_id_qty_alipay_njhs_avg','supplier_id_qty_alipay_njhs_std']
test_all = pd.read_csv('/Applications/dev/tmp/test_all.csv',header=None)
test_all.columns = ['watch','store_code','item_id','label','pv_ipv_1','pv_uv_1','cart_ipv_1','cart_uv_1','collect_uv_1','num_gmv_1','amt_gmv_1','qty_gmv_1','unum_gmv_1','amt_alipay_1','num_alipay_1','qty_alipay_1','unum_alipay_1','ztc_pv_ipv_1','tbk_pv_ipv_1','ss_pv_ipv_1','jhs_pv_ipv_1','ztc_pv_uv_1','tbk_pv_uv_1','ss_pv_uv_1','jhs_pv_uv_1','num_alipay_njhs_1','amt_alipay_njhs_1','qty_alipay_njhs_1','unum_alipay_njhs_1','pv_ipv_sum_1','pv_uv_sum_1','cart_ipv_sum_1','cart_uv_sum_1','collect_uv_sum_1','num_gmv_sum_1','amt_gmv_sum_1','qty_gmv_sum_1','unum_gmv_sum_1','amt_alipay_sum_1','num_alipay_sum_1','qty_alipay_sum_1','unum_alipay_sum_1','ztc_pv_ipv_sum_1','tbk_pv_ipv_sum_1','ss_pv_ipv_sum_1','jhs_pv_ipv_sum_1','ztc_pv_uv_sum_1','tbk_pv_uv_sum_1','ss_pv_uv_sum_1','jhs_pv_uv_sum_1','num_alipay_njhs_sum_1','amt_alipay_njhs_sum_1','qty_alipay_njhs_sum_1','unum_alipay_njhs_sum_1','pv_ipv_2','pv_uv_2','cart_ipv_2','cart_uv_2','collect_uv_2','num_gmv_2','amt_gmv_2','qty_gmv_2','unum_gmv_2','amt_alipay_2','num_alipay_2','qty_alipay_2','unum_alipay_2','ztc_pv_ipv_2','tbk_pv_ipv_2','ss_pv_ipv_2','jhs_pv_ipv_2','ztc_pv_uv_2','tbk_pv_uv_2','ss_pv_uv_2','jhs_pv_uv_2','num_alipay_njhs_2','amt_alipay_njhs_2','qty_alipay_njhs_2','unum_alipay_njhs_2','pv_ipv_sum_2','pv_uv_sum_2','cart_ipv_sum_2','cart_uv_sum_2','collect_uv_sum_2','num_gmv_sum_2','amt_gmv_sum_2','qty_gmv_sum_2','unum_gmv_sum_2','amt_alipay_sum_2','num_alipay_sum_2','qty_alipay_sum_2','unum_alipay_sum_2','ztc_pv_ipv_sum_2','tbk_pv_ipv_sum_2','ss_pv_ipv_sum_2','jhs_pv_ipv_sum_2','ztc_pv_uv_sum_2','tbk_pv_uv_sum_2','ss_pv_uv_sum_2','jhs_pv_uv_sum_2','num_alipay_njhs_sum_2','amt_alipay_njhs_sum_2','qty_alipay_njhs_sum_2','unum_alipay_njhs_sum_2','pv_ipv_3','pv_uv_3','cart_ipv_3','cart_uv_3','collect_uv_3','num_gmv_3','amt_gmv_3','qty_gmv_3','unum_gmv_3','amt_alipay_3','num_alipay_3','qty_alipay_3','unum_alipay_3','ztc_pv_ipv_3','tbk_pv_ipv_3','ss_pv_ipv_3','jhs_pv_ipv_3','ztc_pv_uv_3','tbk_pv_uv_3','ss_pv_uv_3','jhs_pv_uv_3','num_alipay_njhs_3','amt_alipay_njhs_3','qty_alipay_njhs_3','unum_alipay_njhs_3','pv_ipv_sum_3','pv_uv_sum_3','cart_ipv_sum_3','cart_uv_sum_3','collect_uv_sum_3','num_gmv_sum_3','amt_gmv_sum_3','qty_gmv_sum_3','unum_gmv_sum_3','amt_alipay_sum_3','num_alipay_sum_3','qty_alipay_sum_3','unum_alipay_sum_3','ztc_pv_ipv_sum_3','tbk_pv_ipv_sum_3','ss_pv_ipv_sum_3','jhs_pv_ipv_sum_3','ztc_pv_uv_sum_3','tbk_pv_uv_sum_3','ss_pv_uv_sum_3','jhs_pv_uv_sum_3','num_alipay_njhs_sum_3','amt_alipay_njhs_sum_3','qty_alipay_njhs_sum_3','unum_alipay_njhs_sum_3','pv_ipv_5','pv_uv_5','cart_ipv_5','cart_uv_5','collect_uv_5','num_gmv_5','amt_gmv_5','qty_gmv_5','unum_gmv_5','amt_alipay_5','num_alipay_5','qty_alipay_5','unum_alipay_5','ztc_pv_ipv_5','tbk_pv_ipv_5','ss_pv_ipv_5','jhs_pv_ipv_5','ztc_pv_uv_5','tbk_pv_uv_5','ss_pv_uv_5','jhs_pv_uv_5','num_alipay_njhs_5','amt_alipay_njhs_5','qty_alipay_njhs_5','unum_alipay_njhs_5','pv_ipv_sum_5','pv_uv_sum_5','cart_ipv_sum_5','cart_uv_sum_5','collect_uv_sum_5','num_gmv_sum_5','amt_gmv_sum_5','qty_gmv_sum_5','unum_gmv_sum_5','amt_alipay_sum_5','num_alipay_sum_5','qty_alipay_sum_5','unum_alipay_sum_5','ztc_pv_ipv_sum_5','tbk_pv_ipv_sum_5','ss_pv_ipv_sum_5','jhs_pv_ipv_sum_5','ztc_pv_uv_sum_5','tbk_pv_uv_sum_5','ss_pv_uv_sum_5','jhs_pv_uv_sum_5','num_alipay_njhs_sum_5','amt_alipay_njhs_sum_5','qty_alipay_njhs_sum_5','unum_alipay_njhs_sum_5','pv_ipv_7','pv_uv_7','cart_ipv_7','cart_uv_7','collect_uv_7','num_gmv_7','amt_gmv_7','qty_gmv_7','unum_gmv_7','amt_alipay_7','num_alipay_7','qty_alipay_7','unum_alipay_7','ztc_pv_ipv_7','tbk_pv_ipv_7','ss_pv_ipv_7','jhs_pv_ipv_7','ztc_pv_uv_7','tbk_pv_uv_7','ss_pv_uv_7','jhs_pv_uv_7','num_alipay_njhs_7','amt_alipay_njhs_7','qty_alipay_njhs_7','unum_alipay_njhs_7','pv_ipv_sum_7','pv_uv_sum_7','cart_ipv_sum_7','cart_uv_sum_7','collect_uv_sum_7','num_gmv_sum_7','amt_gmv_sum_7','qty_gmv_sum_7','unum_gmv_sum_7','amt_alipay_sum_7','num_alipay_sum_7','qty_alipay_sum_7','unum_alipay_sum_7','ztc_pv_ipv_sum_7','tbk_pv_ipv_sum_7','ss_pv_ipv_sum_7','jhs_pv_ipv_sum_7','ztc_pv_uv_sum_7','tbk_pv_uv_sum_7','ss_pv_uv_sum_7','jhs_pv_uv_sum_7','num_alipay_njhs_sum_7','amt_alipay_njhs_sum_7','qty_alipay_njhs_sum_7','unum_alipay_njhs_sum_7','pv_ipv_9','pv_uv_9','cart_ipv_9','cart_uv_9','collect_uv_9','num_gmv_9','amt_gmv_9','qty_gmv_9','unum_gmv_9','amt_alipay_9','num_alipay_9','qty_alipay_9','unum_alipay_9','ztc_pv_ipv_9','tbk_pv_ipv_9','ss_pv_ipv_9','jhs_pv_ipv_9','ztc_pv_uv_9','tbk_pv_uv_9','ss_pv_uv_9','jhs_pv_uv_9','num_alipay_njhs_9','amt_alipay_njhs_9','qty_alipay_njhs_9','unum_alipay_njhs_9','pv_ipv_sum_9','pv_uv_sum_9','cart_ipv_sum_9','cart_uv_sum_9','collect_uv_sum_9','num_gmv_sum_9','amt_gmv_sum_9','qty_gmv_sum_9','unum_gmv_sum_9','amt_alipay_sum_9','num_alipay_sum_9','qty_alipay_sum_9','unum_alipay_sum_9','ztc_pv_ipv_sum_9','tbk_pv_ipv_sum_9','ss_pv_ipv_sum_9','jhs_pv_ipv_sum_9','ztc_pv_uv_sum_9','tbk_pv_uv_sum_9','ss_pv_uv_sum_9','jhs_pv_uv_sum_9','num_alipay_njhs_sum_9','amt_alipay_njhs_sum_9','qty_alipay_njhs_sum_9','unum_alipay_njhs_sum_9','pv_ipv_11','pv_uv_11','cart_ipv_11','cart_uv_11','collect_uv_11','num_gmv_11','amt_gmv_11','qty_gmv_11','unum_gmv_11','amt_alipay_11','num_alipay_11','qty_alipay_11','unum_alipay_11','ztc_pv_ipv_11','tbk_pv_ipv_11','ss_pv_ipv_11','jhs_pv_ipv_11','ztc_pv_uv_11','tbk_pv_uv_11','ss_pv_uv_11','jhs_pv_uv_11','num_alipay_njhs_11','amt_alipay_njhs_11','qty_alipay_njhs_11','unum_alipay_njhs_11','pv_ipv_sum_11','pv_uv_sum_11','cart_ipv_sum_11','cart_uv_sum_11','collect_uv_sum_11','num_gmv_sum_11','amt_gmv_sum_11','qty_gmv_sum_11','unum_gmv_sum_11','amt_alipay_sum_11','num_alipay_sum_11','qty_alipay_sum_11','unum_alipay_sum_11','ztc_pv_ipv_sum_11','tbk_pv_ipv_sum_11','ss_pv_ipv_sum_11','jhs_pv_ipv_sum_11','ztc_pv_uv_sum_11','tbk_pv_uv_sum_11','ss_pv_uv_sum_11','jhs_pv_uv_sum_11','num_alipay_njhs_sum_11','amt_alipay_njhs_sum_11','qty_alipay_njhs_sum_11','unum_alipay_njhs_sum_11','pv_ipv_14','pv_uv_14','cart_ipv_14','cart_uv_14','collect_uv_14','num_gmv_14','amt_gmv_14','qty_gmv_14','unum_gmv_14','amt_alipay_14','num_alipay_14','qty_alipay_14','unum_alipay_14','ztc_pv_ipv_14','tbk_pv_ipv_14','ss_pv_ipv_14','jhs_pv_ipv_14','ztc_pv_uv_14','tbk_pv_uv_14','ss_pv_uv_14','jhs_pv_uv_14','num_alipay_njhs_14','amt_alipay_njhs_14','qty_alipay_njhs_14','unum_alipay_njhs_14','pv_ipv_sum_14','pv_uv_sum_14','cart_ipv_sum_14','cart_uv_sum_14','collect_uv_sum_14','num_gmv_sum_14','amt_gmv_sum_14','qty_gmv_sum_14','unum_gmv_sum_14','amt_alipay_sum_14','num_alipay_sum_14','qty_alipay_sum_14','unum_alipay_sum_14','ztc_pv_ipv_sum_14','tbk_pv_ipv_sum_14','ss_pv_ipv_sum_14','jhs_pv_ipv_sum_14','ztc_pv_uv_sum_14','tbk_pv_uv_sum_14','ss_pv_uv_sum_14','jhs_pv_uv_sum_14','num_alipay_njhs_sum_14','amt_alipay_njhs_sum_14','qty_alipay_njhs_sum_14','unum_alipay_njhs_sum_14','max_qty_alipay_njhs','min_qty_alipay_njhs','std_qty_alipay_njhs','cate_id_qty_alipay_njhs_sum','cate_id_qty_alipay_njhs_avg','cate_id_qty_alipay_njhs_std','cate_level_id_qty_alipay_njhs_sum','cate_level_id_qty_alipay_njhs_avg','cate_level_id_qty_alipay_njhs_std','brand_id_qty_alipay_njhs_sum','brand_id_qty_alipay_njhs_avg','brand_id_qty_alipay_njhs_std','supplier_id_qty_alipay_njhs_sum','supplier_id_qty_alipay_njhs_avg','supplier_id_qty_alipay_njhs_std']
train_fencang = pd.read_csv('/Applications/dev/tmp/train_fencang.csv',header=None)
train_fencang.columns = ['watch','store_code','item_id','label','pv_ipv_1','pv_uv_1','cart_ipv_1','cart_uv_1','collect_uv_1','num_gmv_1','amt_gmv_1','qty_gmv_1','unum_gmv_1','amt_alipay_1','num_alipay_1','qty_alipay_1','unum_alipay_1','ztc_pv_ipv_1','tbk_pv_ipv_1','ss_pv_ipv_1','jhs_pv_ipv_1','ztc_pv_uv_1','tbk_pv_uv_1','ss_pv_uv_1','jhs_pv_uv_1','num_alipay_njhs_1','amt_alipay_njhs_1','qty_alipay_njhs_1','unum_alipay_njhs_1','pv_ipv_sum_1','pv_uv_sum_1','cart_ipv_sum_1','cart_uv_sum_1','collect_uv_sum_1','num_gmv_sum_1','amt_gmv_sum_1','qty_gmv_sum_1','unum_gmv_sum_1','amt_alipay_sum_1','num_alipay_sum_1','qty_alipay_sum_1','unum_alipay_sum_1','ztc_pv_ipv_sum_1','tbk_pv_ipv_sum_1','ss_pv_ipv_sum_1','jhs_pv_ipv_sum_1','ztc_pv_uv_sum_1','tbk_pv_uv_sum_1','ss_pv_uv_sum_1','jhs_pv_uv_sum_1','num_alipay_njhs_sum_1','amt_alipay_njhs_sum_1','qty_alipay_njhs_sum_1','unum_alipay_njhs_sum_1','pv_ipv_2','pv_uv_2','cart_ipv_2','cart_uv_2','collect_uv_2','num_gmv_2','amt_gmv_2','qty_gmv_2','unum_gmv_2','amt_alipay_2','num_alipay_2','qty_alipay_2','unum_alipay_2','ztc_pv_ipv_2','tbk_pv_ipv_2','ss_pv_ipv_2','jhs_pv_ipv_2','ztc_pv_uv_2','tbk_pv_uv_2','ss_pv_uv_2','jhs_pv_uv_2','num_alipay_njhs_2','amt_alipay_njhs_2','qty_alipay_njhs_2','unum_alipay_njhs_2','pv_ipv_sum_2','pv_uv_sum_2','cart_ipv_sum_2','cart_uv_sum_2','collect_uv_sum_2','num_gmv_sum_2','amt_gmv_sum_2','qty_gmv_sum_2','unum_gmv_sum_2','amt_alipay_sum_2','num_alipay_sum_2','qty_alipay_sum_2','unum_alipay_sum_2','ztc_pv_ipv_sum_2','tbk_pv_ipv_sum_2','ss_pv_ipv_sum_2','jhs_pv_ipv_sum_2','ztc_pv_uv_sum_2','tbk_pv_uv_sum_2','ss_pv_uv_sum_2','jhs_pv_uv_sum_2','num_alipay_njhs_sum_2','amt_alipay_njhs_sum_2','qty_alipay_njhs_sum_2','unum_alipay_njhs_sum_2','pv_ipv_3','pv_uv_3','cart_ipv_3','cart_uv_3','collect_uv_3','num_gmv_3','amt_gmv_3','qty_gmv_3','unum_gmv_3','amt_alipay_3','num_alipay_3','qty_alipay_3','unum_alipay_3','ztc_pv_ipv_3','tbk_pv_ipv_3','ss_pv_ipv_3','jhs_pv_ipv_3','ztc_pv_uv_3','tbk_pv_uv_3','ss_pv_uv_3','jhs_pv_uv_3','num_alipay_njhs_3','amt_alipay_njhs_3','qty_alipay_njhs_3','unum_alipay_njhs_3','pv_ipv_sum_3','pv_uv_sum_3','cart_ipv_sum_3','cart_uv_sum_3','collect_uv_sum_3','num_gmv_sum_3','amt_gmv_sum_3','qty_gmv_sum_3','unum_gmv_sum_3','amt_alipay_sum_3','num_alipay_sum_3','qty_alipay_sum_3','unum_alipay_sum_3','ztc_pv_ipv_sum_3','tbk_pv_ipv_sum_3','ss_pv_ipv_sum_3','jhs_pv_ipv_sum_3','ztc_pv_uv_sum_3','tbk_pv_uv_sum_3','ss_pv_uv_sum_3','jhs_pv_uv_sum_3','num_alipay_njhs_sum_3','amt_alipay_njhs_sum_3','qty_alipay_njhs_sum_3','unum_alipay_njhs_sum_3','pv_ipv_5','pv_uv_5','cart_ipv_5','cart_uv_5','collect_uv_5','num_gmv_5','amt_gmv_5','qty_gmv_5','unum_gmv_5','amt_alipay_5','num_alipay_5','qty_alipay_5','unum_alipay_5','ztc_pv_ipv_5','tbk_pv_ipv_5','ss_pv_ipv_5','jhs_pv_ipv_5','ztc_pv_uv_5','tbk_pv_uv_5','ss_pv_uv_5','jhs_pv_uv_5','num_alipay_njhs_5','amt_alipay_njhs_5','qty_alipay_njhs_5','unum_alipay_njhs_5','pv_ipv_sum_5','pv_uv_sum_5','cart_ipv_sum_5','cart_uv_sum_5','collect_uv_sum_5','num_gmv_sum_5','amt_gmv_sum_5','qty_gmv_sum_5','unum_gmv_sum_5','amt_alipay_sum_5','num_alipay_sum_5','qty_alipay_sum_5','unum_alipay_sum_5','ztc_pv_ipv_sum_5','tbk_pv_ipv_sum_5','ss_pv_ipv_sum_5','jhs_pv_ipv_sum_5','ztc_pv_uv_sum_5','tbk_pv_uv_sum_5','ss_pv_uv_sum_5','jhs_pv_uv_sum_5','num_alipay_njhs_sum_5','amt_alipay_njhs_sum_5','qty_alipay_njhs_sum_5','unum_alipay_njhs_sum_5','pv_ipv_7','pv_uv_7','cart_ipv_7','cart_uv_7','collect_uv_7','num_gmv_7','amt_gmv_7','qty_gmv_7','unum_gmv_7','amt_alipay_7','num_alipay_7','qty_alipay_7','unum_alipay_7','ztc_pv_ipv_7','tbk_pv_ipv_7','ss_pv_ipv_7','jhs_pv_ipv_7','ztc_pv_uv_7','tbk_pv_uv_7','ss_pv_uv_7','jhs_pv_uv_7','num_alipay_njhs_7','amt_alipay_njhs_7','qty_alipay_njhs_7','unum_alipay_njhs_7','pv_ipv_sum_7','pv_uv_sum_7','cart_ipv_sum_7','cart_uv_sum_7','collect_uv_sum_7','num_gmv_sum_7','amt_gmv_sum_7','qty_gmv_sum_7','unum_gmv_sum_7','amt_alipay_sum_7','num_alipay_sum_7','qty_alipay_sum_7','unum_alipay_sum_7','ztc_pv_ipv_sum_7','tbk_pv_ipv_sum_7','ss_pv_ipv_sum_7','jhs_pv_ipv_sum_7','ztc_pv_uv_sum_7','tbk_pv_uv_sum_7','ss_pv_uv_sum_7','jhs_pv_uv_sum_7','num_alipay_njhs_sum_7','amt_alipay_njhs_sum_7','qty_alipay_njhs_sum_7','unum_alipay_njhs_sum_7','pv_ipv_9','pv_uv_9','cart_ipv_9','cart_uv_9','collect_uv_9','num_gmv_9','amt_gmv_9','qty_gmv_9','unum_gmv_9','amt_alipay_9','num_alipay_9','qty_alipay_9','unum_alipay_9','ztc_pv_ipv_9','tbk_pv_ipv_9','ss_pv_ipv_9','jhs_pv_ipv_9','ztc_pv_uv_9','tbk_pv_uv_9','ss_pv_uv_9','jhs_pv_uv_9','num_alipay_njhs_9','amt_alipay_njhs_9','qty_alipay_njhs_9','unum_alipay_njhs_9','pv_ipv_sum_9','pv_uv_sum_9','cart_ipv_sum_9','cart_uv_sum_9','collect_uv_sum_9','num_gmv_sum_9','amt_gmv_sum_9','qty_gmv_sum_9','unum_gmv_sum_9','amt_alipay_sum_9','num_alipay_sum_9','qty_alipay_sum_9','unum_alipay_sum_9','ztc_pv_ipv_sum_9','tbk_pv_ipv_sum_9','ss_pv_ipv_sum_9','jhs_pv_ipv_sum_9','ztc_pv_uv_sum_9','tbk_pv_uv_sum_9','ss_pv_uv_sum_9','jhs_pv_uv_sum_9','num_alipay_njhs_sum_9','amt_alipay_njhs_sum_9','qty_alipay_njhs_sum_9','unum_alipay_njhs_sum_9','pv_ipv_11','pv_uv_11','cart_ipv_11','cart_uv_11','collect_uv_11','num_gmv_11','amt_gmv_11','qty_gmv_11','unum_gmv_11','amt_alipay_11','num_alipay_11','qty_alipay_11','unum_alipay_11','ztc_pv_ipv_11','tbk_pv_ipv_11','ss_pv_ipv_11','jhs_pv_ipv_11','ztc_pv_uv_11','tbk_pv_uv_11','ss_pv_uv_11','jhs_pv_uv_11','num_alipay_njhs_11','amt_alipay_njhs_11','qty_alipay_njhs_11','unum_alipay_njhs_11','pv_ipv_sum_11','pv_uv_sum_11','cart_ipv_sum_11','cart_uv_sum_11','collect_uv_sum_11','num_gmv_sum_11','amt_gmv_sum_11','qty_gmv_sum_11','unum_gmv_sum_11','amt_alipay_sum_11','num_alipay_sum_11','qty_alipay_sum_11','unum_alipay_sum_11','ztc_pv_ipv_sum_11','tbk_pv_ipv_sum_11','ss_pv_ipv_sum_11','jhs_pv_ipv_sum_11','ztc_pv_uv_sum_11','tbk_pv_uv_sum_11','ss_pv_uv_sum_11','jhs_pv_uv_sum_11','num_alipay_njhs_sum_11','amt_alipay_njhs_sum_11','qty_alipay_njhs_sum_11','unum_alipay_njhs_sum_11','pv_ipv_14','pv_uv_14','cart_ipv_14','cart_uv_14','collect_uv_14','num_gmv_14','amt_gmv_14','qty_gmv_14','unum_gmv_14','amt_alipay_14','num_alipay_14','qty_alipay_14','unum_alipay_14','ztc_pv_ipv_14','tbk_pv_ipv_14','ss_pv_ipv_14','jhs_pv_ipv_14','ztc_pv_uv_14','tbk_pv_uv_14','ss_pv_uv_14','jhs_pv_uv_14','num_alipay_njhs_14','amt_alipay_njhs_14','qty_alipay_njhs_14','unum_alipay_njhs_14','pv_ipv_sum_14','pv_uv_sum_14','cart_ipv_sum_14','cart_uv_sum_14','collect_uv_sum_14','num_gmv_sum_14','amt_gmv_sum_14','qty_gmv_sum_14','unum_gmv_sum_14','amt_alipay_sum_14','num_alipay_sum_14','qty_alipay_sum_14','unum_alipay_sum_14','ztc_pv_ipv_sum_14','tbk_pv_ipv_sum_14','ss_pv_ipv_sum_14','jhs_pv_ipv_sum_14','ztc_pv_uv_sum_14','tbk_pv_uv_sum_14','ss_pv_uv_sum_14','jhs_pv_uv_sum_14','num_alipay_njhs_sum_14','amt_alipay_njhs_sum_14','qty_alipay_njhs_sum_14','unum_alipay_njhs_sum_14','max_qty_alipay_njhs','min_qty_alipay_njhs','std_qty_alipay_njhs','cate_id_qty_alipay_njhs_sum','cate_id_qty_alipay_njhs_avg','cate_id_qty_alipay_njhs_std','cate_level_id_qty_alipay_njhs_sum','cate_level_id_qty_alipay_njhs_avg','cate_level_id_qty_alipay_njhs_std','brand_id_qty_alipay_njhs_sum','brand_id_qty_alipay_njhs_avg','brand_id_qty_alipay_njhs_std','supplier_id_qty_alipay_njhs_sum','supplier_id_qty_alipay_njhs_avg','supplier_id_qty_alipay_njhs_std']
test_fencang = pd.read_csv('/Applications/dev/tmp/test_fencang.csv',header=None)
test_fencang.columns = ['watch','store_code','item_id','label','pv_ipv_1','pv_uv_1','cart_ipv_1','cart_uv_1','collect_uv_1','num_gmv_1','amt_gmv_1','qty_gmv_1','unum_gmv_1','amt_alipay_1','num_alipay_1','qty_alipay_1','unum_alipay_1','ztc_pv_ipv_1','tbk_pv_ipv_1','ss_pv_ipv_1','jhs_pv_ipv_1','ztc_pv_uv_1','tbk_pv_uv_1','ss_pv_uv_1','jhs_pv_uv_1','num_alipay_njhs_1','amt_alipay_njhs_1','qty_alipay_njhs_1','unum_alipay_njhs_1','pv_ipv_sum_1','pv_uv_sum_1','cart_ipv_sum_1','cart_uv_sum_1','collect_uv_sum_1','num_gmv_sum_1','amt_gmv_sum_1','qty_gmv_sum_1','unum_gmv_sum_1','amt_alipay_sum_1','num_alipay_sum_1','qty_alipay_sum_1','unum_alipay_sum_1','ztc_pv_ipv_sum_1','tbk_pv_ipv_sum_1','ss_pv_ipv_sum_1','jhs_pv_ipv_sum_1','ztc_pv_uv_sum_1','tbk_pv_uv_sum_1','ss_pv_uv_sum_1','jhs_pv_uv_sum_1','num_alipay_njhs_sum_1','amt_alipay_njhs_sum_1','qty_alipay_njhs_sum_1','unum_alipay_njhs_sum_1','pv_ipv_2','pv_uv_2','cart_ipv_2','cart_uv_2','collect_uv_2','num_gmv_2','amt_gmv_2','qty_gmv_2','unum_gmv_2','amt_alipay_2','num_alipay_2','qty_alipay_2','unum_alipay_2','ztc_pv_ipv_2','tbk_pv_ipv_2','ss_pv_ipv_2','jhs_pv_ipv_2','ztc_pv_uv_2','tbk_pv_uv_2','ss_pv_uv_2','jhs_pv_uv_2','num_alipay_njhs_2','amt_alipay_njhs_2','qty_alipay_njhs_2','unum_alipay_njhs_2','pv_ipv_sum_2','pv_uv_sum_2','cart_ipv_sum_2','cart_uv_sum_2','collect_uv_sum_2','num_gmv_sum_2','amt_gmv_sum_2','qty_gmv_sum_2','unum_gmv_sum_2','amt_alipay_sum_2','num_alipay_sum_2','qty_alipay_sum_2','unum_alipay_sum_2','ztc_pv_ipv_sum_2','tbk_pv_ipv_sum_2','ss_pv_ipv_sum_2','jhs_pv_ipv_sum_2','ztc_pv_uv_sum_2','tbk_pv_uv_sum_2','ss_pv_uv_sum_2','jhs_pv_uv_sum_2','num_alipay_njhs_sum_2','amt_alipay_njhs_sum_2','qty_alipay_njhs_sum_2','unum_alipay_njhs_sum_2','pv_ipv_3','pv_uv_3','cart_ipv_3','cart_uv_3','collect_uv_3','num_gmv_3','amt_gmv_3','qty_gmv_3','unum_gmv_3','amt_alipay_3','num_alipay_3','qty_alipay_3','unum_alipay_3','ztc_pv_ipv_3','tbk_pv_ipv_3','ss_pv_ipv_3','jhs_pv_ipv_3','ztc_pv_uv_3','tbk_pv_uv_3','ss_pv_uv_3','jhs_pv_uv_3','num_alipay_njhs_3','amt_alipay_njhs_3','qty_alipay_njhs_3','unum_alipay_njhs_3','pv_ipv_sum_3','pv_uv_sum_3','cart_ipv_sum_3','cart_uv_sum_3','collect_uv_sum_3','num_gmv_sum_3','amt_gmv_sum_3','qty_gmv_sum_3','unum_gmv_sum_3','amt_alipay_sum_3','num_alipay_sum_3','qty_alipay_sum_3','unum_alipay_sum_3','ztc_pv_ipv_sum_3','tbk_pv_ipv_sum_3','ss_pv_ipv_sum_3','jhs_pv_ipv_sum_3','ztc_pv_uv_sum_3','tbk_pv_uv_sum_3','ss_pv_uv_sum_3','jhs_pv_uv_sum_3','num_alipay_njhs_sum_3','amt_alipay_njhs_sum_3','qty_alipay_njhs_sum_3','unum_alipay_njhs_sum_3','pv_ipv_5','pv_uv_5','cart_ipv_5','cart_uv_5','collect_uv_5','num_gmv_5','amt_gmv_5','qty_gmv_5','unum_gmv_5','amt_alipay_5','num_alipay_5','qty_alipay_5','unum_alipay_5','ztc_pv_ipv_5','tbk_pv_ipv_5','ss_pv_ipv_5','jhs_pv_ipv_5','ztc_pv_uv_5','tbk_pv_uv_5','ss_pv_uv_5','jhs_pv_uv_5','num_alipay_njhs_5','amt_alipay_njhs_5','qty_alipay_njhs_5','unum_alipay_njhs_5','pv_ipv_sum_5','pv_uv_sum_5','cart_ipv_sum_5','cart_uv_sum_5','collect_uv_sum_5','num_gmv_sum_5','amt_gmv_sum_5','qty_gmv_sum_5','unum_gmv_sum_5','amt_alipay_sum_5','num_alipay_sum_5','qty_alipay_sum_5','unum_alipay_sum_5','ztc_pv_ipv_sum_5','tbk_pv_ipv_sum_5','ss_pv_ipv_sum_5','jhs_pv_ipv_sum_5','ztc_pv_uv_sum_5','tbk_pv_uv_sum_5','ss_pv_uv_sum_5','jhs_pv_uv_sum_5','num_alipay_njhs_sum_5','amt_alipay_njhs_sum_5','qty_alipay_njhs_sum_5','unum_alipay_njhs_sum_5','pv_ipv_7','pv_uv_7','cart_ipv_7','cart_uv_7','collect_uv_7','num_gmv_7','amt_gmv_7','qty_gmv_7','unum_gmv_7','amt_alipay_7','num_alipay_7','qty_alipay_7','unum_alipay_7','ztc_pv_ipv_7','tbk_pv_ipv_7','ss_pv_ipv_7','jhs_pv_ipv_7','ztc_pv_uv_7','tbk_pv_uv_7','ss_pv_uv_7','jhs_pv_uv_7','num_alipay_njhs_7','amt_alipay_njhs_7','qty_alipay_njhs_7','unum_alipay_njhs_7','pv_ipv_sum_7','pv_uv_sum_7','cart_ipv_sum_7','cart_uv_sum_7','collect_uv_sum_7','num_gmv_sum_7','amt_gmv_sum_7','qty_gmv_sum_7','unum_gmv_sum_7','amt_alipay_sum_7','num_alipay_sum_7','qty_alipay_sum_7','unum_alipay_sum_7','ztc_pv_ipv_sum_7','tbk_pv_ipv_sum_7','ss_pv_ipv_sum_7','jhs_pv_ipv_sum_7','ztc_pv_uv_sum_7','tbk_pv_uv_sum_7','ss_pv_uv_sum_7','jhs_pv_uv_sum_7','num_alipay_njhs_sum_7','amt_alipay_njhs_sum_7','qty_alipay_njhs_sum_7','unum_alipay_njhs_sum_7','pv_ipv_9','pv_uv_9','cart_ipv_9','cart_uv_9','collect_uv_9','num_gmv_9','amt_gmv_9','qty_gmv_9','unum_gmv_9','amt_alipay_9','num_alipay_9','qty_alipay_9','unum_alipay_9','ztc_pv_ipv_9','tbk_pv_ipv_9','ss_pv_ipv_9','jhs_pv_ipv_9','ztc_pv_uv_9','tbk_pv_uv_9','ss_pv_uv_9','jhs_pv_uv_9','num_alipay_njhs_9','amt_alipay_njhs_9','qty_alipay_njhs_9','unum_alipay_njhs_9','pv_ipv_sum_9','pv_uv_sum_9','cart_ipv_sum_9','cart_uv_sum_9','collect_uv_sum_9','num_gmv_sum_9','amt_gmv_sum_9','qty_gmv_sum_9','unum_gmv_sum_9','amt_alipay_sum_9','num_alipay_sum_9','qty_alipay_sum_9','unum_alipay_sum_9','ztc_pv_ipv_sum_9','tbk_pv_ipv_sum_9','ss_pv_ipv_sum_9','jhs_pv_ipv_sum_9','ztc_pv_uv_sum_9','tbk_pv_uv_sum_9','ss_pv_uv_sum_9','jhs_pv_uv_sum_9','num_alipay_njhs_sum_9','amt_alipay_njhs_sum_9','qty_alipay_njhs_sum_9','unum_alipay_njhs_sum_9','pv_ipv_11','pv_uv_11','cart_ipv_11','cart_uv_11','collect_uv_11','num_gmv_11','amt_gmv_11','qty_gmv_11','unum_gmv_11','amt_alipay_11','num_alipay_11','qty_alipay_11','unum_alipay_11','ztc_pv_ipv_11','tbk_pv_ipv_11','ss_pv_ipv_11','jhs_pv_ipv_11','ztc_pv_uv_11','tbk_pv_uv_11','ss_pv_uv_11','jhs_pv_uv_11','num_alipay_njhs_11','amt_alipay_njhs_11','qty_alipay_njhs_11','unum_alipay_njhs_11','pv_ipv_sum_11','pv_uv_sum_11','cart_ipv_sum_11','cart_uv_sum_11','collect_uv_sum_11','num_gmv_sum_11','amt_gmv_sum_11','qty_gmv_sum_11','unum_gmv_sum_11','amt_alipay_sum_11','num_alipay_sum_11','qty_alipay_sum_11','unum_alipay_sum_11','ztc_pv_ipv_sum_11','tbk_pv_ipv_sum_11','ss_pv_ipv_sum_11','jhs_pv_ipv_sum_11','ztc_pv_uv_sum_11','tbk_pv_uv_sum_11','ss_pv_uv_sum_11','jhs_pv_uv_sum_11','num_alipay_njhs_sum_11','amt_alipay_njhs_sum_11','qty_alipay_njhs_sum_11','unum_alipay_njhs_sum_11','pv_ipv_14','pv_uv_14','cart_ipv_14','cart_uv_14','collect_uv_14','num_gmv_14','amt_gmv_14','qty_gmv_14','unum_gmv_14','amt_alipay_14','num_alipay_14','qty_alipay_14','unum_alipay_14','ztc_pv_ipv_14','tbk_pv_ipv_14','ss_pv_ipv_14','jhs_pv_ipv_14','ztc_pv_uv_14','tbk_pv_uv_14','ss_pv_uv_14','jhs_pv_uv_14','num_alipay_njhs_14','amt_alipay_njhs_14','qty_alipay_njhs_14','unum_alipay_njhs_14','pv_ipv_sum_14','pv_uv_sum_14','cart_ipv_sum_14','cart_uv_sum_14','collect_uv_sum_14','num_gmv_sum_14','amt_gmv_sum_14','qty_gmv_sum_14','unum_gmv_sum_14','amt_alipay_sum_14','num_alipay_sum_14','qty_alipay_sum_14','unum_alipay_sum_14','ztc_pv_ipv_sum_14','tbk_pv_ipv_sum_14','ss_pv_ipv_sum_14','jhs_pv_ipv_sum_14','ztc_pv_uv_sum_14','tbk_pv_uv_sum_14','ss_pv_uv_sum_14','jhs_pv_uv_sum_14','num_alipay_njhs_sum_14','amt_alipay_njhs_sum_14','qty_alipay_njhs_sum_14','unum_alipay_njhs_sum_14','max_qty_alipay_njhs','min_qty_alipay_njhs','std_qty_alipay_njhs','cate_id_qty_alipay_njhs_sum','cate_id_qty_alipay_njhs_avg','cate_id_qty_alipay_njhs_std','cate_level_id_qty_alipay_njhs_sum','cate_level_id_qty_alipay_njhs_avg','cate_level_id_qty_alipay_njhs_std','brand_id_qty_alipay_njhs_sum','brand_id_qty_alipay_njhs_avg','brand_id_qty_alipay_njhs_std','supplier_id_qty_alipay_njhs_sum','supplier_id_qty_alipay_njhs_avg','supplier_id_qty_alipay_njhs_std']


train_all = train_all.replace("\N",0.0)
test_all = test_all.replace("\N",0.0)
train_fencang = train_fencang.replace("\N",0.0)
test_fencang = test_fencang.replace("\N",0.0)


# In[113]:
# 数据统一化为浮点型
train_all = train_all.apply(lambda x: pd.to_numeric(x, errors='ignore'))
test_all = test_all.apply(lambda x: pd.to_numeric(x, errors='ignore'))
train_fencang = train_fencang.apply(lambda x: pd.to_numeric(x, errors='ignore'))
test_fencang = test_fencang.apply(lambda x: pd.to_numeric(x, errors='ignore'))



# In[115]:
# 数据备份
train_all.to_csv('../data/train_all.csv',index=None)
test_all.to_csv('../data/test_all.csv',index=None)
train_fencang.to_csv('../data/train_fencang.csv',index=None)
test_fencang.to_csv('../data/test_fencang.csv',index=None)

# 处理合并数据
train_all_fencang = pd.concat([train_all,train_fencang])
train_all_fencang.store_code = train_all_fencang.store_code.apply(lambda x: str(x))



# 加载代价数据
a_b = pd.read_csv('../data/config2.csv')

data_train = train_all_fencang

# 映射处理
hashmap = {'1':'1','2':'2','3':'3','4':'4','5':'5','all':'6'}
data_train.store_code = data_train.store_code.apply(lambda x:hashmap[x])
a_b.store_code = a_b.store_code.apply(lambda x:hashmap[x])

# 融合原始数据与代价
data_train = pd.merge(data_train,a_b,on=['item_id','store_code'])


# 计算代价
def cal_cost(y,pred,a,b):
    cost = 0
    nb_sample = len(y)
    for i in range(nb_sample):
        if pred[i]>y[i]:
            cost += b[i]*(pred[i]-y[i])
        else:
            cost += a[i]*(y[i]-pred[i])
    return nb_sample,cost

# gbrt模型
def model_gbrt(data_train,store_code,loss='lad',learning_rate=0.01,n_estimators=400,subsample=0.75,max_depth=6, max_features=0.75):
    starttime = datetime.datetime.now() 
    a_b = pd.read_csv('../data/config2.csv')
    data = data_train
    data = data[data.store_code== store_code]
    val = data[data.watch==1]
    val_a_b = val[['item_id','store_code','a','b']]
    val_y = val.label
    val_x = val.drop(['label','watch','item_id','store_code','a','b','a_b'],axis=1)
    train = data[(data.watch!=1)&(data.watch!=0)]
    train_y = train.label
    a = list(train.a)
    b = list(train.b)
    train_weight = []
    for i in range(len(a)):
        train_weight.append(min(a[i],b[i]))
    train_weight = np.array(train_weight)

    train_x = train.drop(['label','watch','item_id','store_code','a','b','a_b'],axis=1)
    val_x = val_x.fillna(val_x.median(),inplace=True)
    train_x = train_x.fillna(train_x.median(),inplace=True)
    model = GradientBoostingRegressor(loss=loss,learning_rate=learning_rate,n_estimators=n_estimators,subsample=subsample,max_depth=max_depth,random_state=1024, max_features=max_features)
    model.fit(train_x,train_y, sample_weight=train_weight)
    val_a_b['pred'] = model.predict(val_x)
    val_a_b['y'] = val_y
    cost = cal_cost(val_y.values,val_a_b.pred.values,val_a_b.a.values,val_a_b.b.values)
    val_a_b.to_csv('val_{0}.csv'.format(store_code),index=None)
    print "used time: " + str((endtime - starttime).seconds)    + " s" 


# xgb模型
def model_xgb(data_train,store_code,num_boost_round = 500,eta = 0.008,max_depth = 7,subsample = 0.7,colsample_bytree=0.7):
    starttime = datetime.datetime.now() 
    data = data_train
    data = data[data.store_code== store_code]
    val = data[data.watch==1]
    test = data[data.watch==0]
    test_a_b = test[['item_id','store_code','a','b']]
    test_x = test.drop(['label','watch','item_id','store_code','a','b','a_b'],axis=1)
    val_a_b = val[['item_id','store_code','a','b']]
    val_y = val.label
    val_x = val.drop(['label','watch','item_id','store_code','a','b','a_b'],axis=1)
    train = data[(data.watch!=1)&(data.watch!=0)]
    train_x = train.drop(['label','watch','item_id','store_code','a','b','a_b'],axis=1)
    train_y = train.label
    a = list(train.a)
    b = list(train.b)
    train_weight = []
    for i in range(len(a)):
        train_weight.append(min(a[i],b[i]))
    train_weight = np.array(train_weight)


    dtrain = xgb.DMatrix(train_x,label=train_y,weight=train_weight)
    dval = xgb.DMatrix(val_x)
    dtest = xgb.DMatrix(test_x)
# cv调参
    params={
            'booster':'gbtree',
            'objective': 'reg:linear',
            'eval_metric': 'rmse',  # 回归问题，默认值是rmse，对于分类问题，默认值是error。
            'max_depth':max_depth,  # 3-10
            'lambda':100,  # 控制XGBoost的正则化部分
            'subsample':subsample,  # 0.5-1
            'colsample_bytree':colsample_bytree,  # 0.5-1
            'eta': eta,  # 0.01-0.2
            'seed':1024,
            'nthread':8
            #'gamma':1 # 这个参数的值越大，算法越保守。这个参数的值和损失函数息息相关，所以是需要调整的
        }

    watchlist  = [(dtrain,'train')]
    model = xgb.train(params,dtrain,num_boost_round=num_boost_round,evals=watchlist)        
    #predict val set
    val_a_b['pred'] = model.predict(dval)
    val_a_b['y'] = val_y
    cost = cal_cost(val_y.values,val_a_b.pred.values,val_a_b.a.values,val_a_b.b.values)
    val_a_b.to_csv('val_xgb_{0}.csv'.format(store_code),index=None)
    print "used time: " + str((endtime - starttime).seconds)    + " s" 





def model_rf(data_train,store_code,n_estimators=500,max_depth=7,max_features=0.8):
    starttime = datetime.datetime.now() 
    data = data_train
    data = data[data.store_code== store_code]
    val = data[data.watch==1]
    test = data[data.watch==0]
    test_a_b = test[['item_id','store_code','a','b']]
    test_x = test.drop(['label','watch','item_id','store_code','a','b','a_b'],axis=1)
    val_a_b = val[['item_id','store_code','a','b']]
    val_y = val.label
    val_x = val.drop(['label','watch','item_id','store_code','a','b','a_b'],axis=1)
    train = data[(data.watch!=1)&(data.watch!=0)]
    train_x = train.drop(['label','watch','item_id','store_code','a','b','a_b'],axis=1)
    train_y = train.label
    a = list(train.a)
    b = list(train.b)
    train_weight = []
    for i in range(len(a)):
        train_weight.append(min(a[i],b[i]))
    train_weight = np.array(train_weight)
    model = RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth,max_features=max_features,n_jobs=-1,random_state=1024)

    #train
    model.fit(train_x,train_y, sample_weight=train_weight)
    val_a_b['pred'] = model.predict(val_x)
    val_a_b['y'] = val_y
    cost = cal_cost(val_y.values,val_a_b.pred.values,val_a_b.a.values,val_a_b.b.values)
    val_a_b.to_csv('val_rf_{0}.csv'.format(store_code),index=None)

    print "used time: " + str((endtime - starttime).seconds)    + " s" 




if __name__ == "__main__":
    model_gbrt(data_train,'1',max_depth=6)
    model_gbrt(data_train,'2',max_depth=5)
    model_gbrt(data_train,'3',max_depth=5)
    model_gbrt(data_train,'4')
    model_gbrt(data_train,'5')
    model_gbrt(data_train,'6')
    model_xgb(data_train,'1')
    model_xgb(data_train,'2',num_boost_round=400)
    model_xgb(data_train,'3',eta=0.08)
    model_xgb(data_train,'4',eta=0.08,num_boost_round=400)
    model_xgb(data_train,'5')
    model_xgb(data_train,'6',num_boost_round=400)
    model_rf(data_train,'1',max_depth=7)
    model_rf(data_train,'2',max_depth=5)
    model_rf(data_train,'3',max_depth=7,0.7)
    model_rf(data_train,'4',4,0.6)
    model_rf(data_train,'5'5,0.6)
    model_rf(data_train,'6')




# 合并文件














