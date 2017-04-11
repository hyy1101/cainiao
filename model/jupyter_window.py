
# coding: utf-8

# In[513]:

import pandas as pd
import numpy as np
import datetime,time


# In[26]:

import xgboost as xgb


# In[514]:

import re
from datetime import date


# In[470]:

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor


# In[28]:

import os,cPickle,random


# In[83]:

config = pd.read_csv('../data/config2.csv',header=None)


# In[84]:

train_all = pd.read_csv('/Applications/dev/tmp/train_all.csv',header=None)


# In[21]:

train_all.head(5)


# In[282]:

train_all = pd.read_csv('/Applications/dev/tmp/train_all.csv',header=None)
train_all.columns = ['watch','store_code','item_id','label','pv_ipv_1','pv_uv_1','cart_ipv_1','cart_uv_1','collect_uv_1','num_gmv_1','amt_gmv_1','qty_gmv_1','unum_gmv_1','amt_alipay_1','num_alipay_1','qty_alipay_1','unum_alipay_1','ztc_pv_ipv_1','tbk_pv_ipv_1','ss_pv_ipv_1','jhs_pv_ipv_1','ztc_pv_uv_1','tbk_pv_uv_1','ss_pv_uv_1','jhs_pv_uv_1','num_alipay_njhs_1','amt_alipay_njhs_1','qty_alipay_njhs_1','unum_alipay_njhs_1','pv_ipv_sum_1','pv_uv_sum_1','cart_ipv_sum_1','cart_uv_sum_1','collect_uv_sum_1','num_gmv_sum_1','amt_gmv_sum_1','qty_gmv_sum_1','unum_gmv_sum_1','amt_alipay_sum_1','num_alipay_sum_1','qty_alipay_sum_1','unum_alipay_sum_1','ztc_pv_ipv_sum_1','tbk_pv_ipv_sum_1','ss_pv_ipv_sum_1','jhs_pv_ipv_sum_1','ztc_pv_uv_sum_1','tbk_pv_uv_sum_1','ss_pv_uv_sum_1','jhs_pv_uv_sum_1','num_alipay_njhs_sum_1','amt_alipay_njhs_sum_1','qty_alipay_njhs_sum_1','unum_alipay_njhs_sum_1','pv_ipv_2','pv_uv_2','cart_ipv_2','cart_uv_2','collect_uv_2','num_gmv_2','amt_gmv_2','qty_gmv_2','unum_gmv_2','amt_alipay_2','num_alipay_2','qty_alipay_2','unum_alipay_2','ztc_pv_ipv_2','tbk_pv_ipv_2','ss_pv_ipv_2','jhs_pv_ipv_2','ztc_pv_uv_2','tbk_pv_uv_2','ss_pv_uv_2','jhs_pv_uv_2','num_alipay_njhs_2','amt_alipay_njhs_2','qty_alipay_njhs_2','unum_alipay_njhs_2','pv_ipv_sum_2','pv_uv_sum_2','cart_ipv_sum_2','cart_uv_sum_2','collect_uv_sum_2','num_gmv_sum_2','amt_gmv_sum_2','qty_gmv_sum_2','unum_gmv_sum_2','amt_alipay_sum_2','num_alipay_sum_2','qty_alipay_sum_2','unum_alipay_sum_2','ztc_pv_ipv_sum_2','tbk_pv_ipv_sum_2','ss_pv_ipv_sum_2','jhs_pv_ipv_sum_2','ztc_pv_uv_sum_2','tbk_pv_uv_sum_2','ss_pv_uv_sum_2','jhs_pv_uv_sum_2','num_alipay_njhs_sum_2','amt_alipay_njhs_sum_2','qty_alipay_njhs_sum_2','unum_alipay_njhs_sum_2','pv_ipv_3','pv_uv_3','cart_ipv_3','cart_uv_3','collect_uv_3','num_gmv_3','amt_gmv_3','qty_gmv_3','unum_gmv_3','amt_alipay_3','num_alipay_3','qty_alipay_3','unum_alipay_3','ztc_pv_ipv_3','tbk_pv_ipv_3','ss_pv_ipv_3','jhs_pv_ipv_3','ztc_pv_uv_3','tbk_pv_uv_3','ss_pv_uv_3','jhs_pv_uv_3','num_alipay_njhs_3','amt_alipay_njhs_3','qty_alipay_njhs_3','unum_alipay_njhs_3','pv_ipv_sum_3','pv_uv_sum_3','cart_ipv_sum_3','cart_uv_sum_3','collect_uv_sum_3','num_gmv_sum_3','amt_gmv_sum_3','qty_gmv_sum_3','unum_gmv_sum_3','amt_alipay_sum_3','num_alipay_sum_3','qty_alipay_sum_3','unum_alipay_sum_3','ztc_pv_ipv_sum_3','tbk_pv_ipv_sum_3','ss_pv_ipv_sum_3','jhs_pv_ipv_sum_3','ztc_pv_uv_sum_3','tbk_pv_uv_sum_3','ss_pv_uv_sum_3','jhs_pv_uv_sum_3','num_alipay_njhs_sum_3','amt_alipay_njhs_sum_3','qty_alipay_njhs_sum_3','unum_alipay_njhs_sum_3','pv_ipv_5','pv_uv_5','cart_ipv_5','cart_uv_5','collect_uv_5','num_gmv_5','amt_gmv_5','qty_gmv_5','unum_gmv_5','amt_alipay_5','num_alipay_5','qty_alipay_5','unum_alipay_5','ztc_pv_ipv_5','tbk_pv_ipv_5','ss_pv_ipv_5','jhs_pv_ipv_5','ztc_pv_uv_5','tbk_pv_uv_5','ss_pv_uv_5','jhs_pv_uv_5','num_alipay_njhs_5','amt_alipay_njhs_5','qty_alipay_njhs_5','unum_alipay_njhs_5','pv_ipv_sum_5','pv_uv_sum_5','cart_ipv_sum_5','cart_uv_sum_5','collect_uv_sum_5','num_gmv_sum_5','amt_gmv_sum_5','qty_gmv_sum_5','unum_gmv_sum_5','amt_alipay_sum_5','num_alipay_sum_5','qty_alipay_sum_5','unum_alipay_sum_5','ztc_pv_ipv_sum_5','tbk_pv_ipv_sum_5','ss_pv_ipv_sum_5','jhs_pv_ipv_sum_5','ztc_pv_uv_sum_5','tbk_pv_uv_sum_5','ss_pv_uv_sum_5','jhs_pv_uv_sum_5','num_alipay_njhs_sum_5','amt_alipay_njhs_sum_5','qty_alipay_njhs_sum_5','unum_alipay_njhs_sum_5','pv_ipv_7','pv_uv_7','cart_ipv_7','cart_uv_7','collect_uv_7','num_gmv_7','amt_gmv_7','qty_gmv_7','unum_gmv_7','amt_alipay_7','num_alipay_7','qty_alipay_7','unum_alipay_7','ztc_pv_ipv_7','tbk_pv_ipv_7','ss_pv_ipv_7','jhs_pv_ipv_7','ztc_pv_uv_7','tbk_pv_uv_7','ss_pv_uv_7','jhs_pv_uv_7','num_alipay_njhs_7','amt_alipay_njhs_7','qty_alipay_njhs_7','unum_alipay_njhs_7','pv_ipv_sum_7','pv_uv_sum_7','cart_ipv_sum_7','cart_uv_sum_7','collect_uv_sum_7','num_gmv_sum_7','amt_gmv_sum_7','qty_gmv_sum_7','unum_gmv_sum_7','amt_alipay_sum_7','num_alipay_sum_7','qty_alipay_sum_7','unum_alipay_sum_7','ztc_pv_ipv_sum_7','tbk_pv_ipv_sum_7','ss_pv_ipv_sum_7','jhs_pv_ipv_sum_7','ztc_pv_uv_sum_7','tbk_pv_uv_sum_7','ss_pv_uv_sum_7','jhs_pv_uv_sum_7','num_alipay_njhs_sum_7','amt_alipay_njhs_sum_7','qty_alipay_njhs_sum_7','unum_alipay_njhs_sum_7','pv_ipv_9','pv_uv_9','cart_ipv_9','cart_uv_9','collect_uv_9','num_gmv_9','amt_gmv_9','qty_gmv_9','unum_gmv_9','amt_alipay_9','num_alipay_9','qty_alipay_9','unum_alipay_9','ztc_pv_ipv_9','tbk_pv_ipv_9','ss_pv_ipv_9','jhs_pv_ipv_9','ztc_pv_uv_9','tbk_pv_uv_9','ss_pv_uv_9','jhs_pv_uv_9','num_alipay_njhs_9','amt_alipay_njhs_9','qty_alipay_njhs_9','unum_alipay_njhs_9','pv_ipv_sum_9','pv_uv_sum_9','cart_ipv_sum_9','cart_uv_sum_9','collect_uv_sum_9','num_gmv_sum_9','amt_gmv_sum_9','qty_gmv_sum_9','unum_gmv_sum_9','amt_alipay_sum_9','num_alipay_sum_9','qty_alipay_sum_9','unum_alipay_sum_9','ztc_pv_ipv_sum_9','tbk_pv_ipv_sum_9','ss_pv_ipv_sum_9','jhs_pv_ipv_sum_9','ztc_pv_uv_sum_9','tbk_pv_uv_sum_9','ss_pv_uv_sum_9','jhs_pv_uv_sum_9','num_alipay_njhs_sum_9','amt_alipay_njhs_sum_9','qty_alipay_njhs_sum_9','unum_alipay_njhs_sum_9','pv_ipv_11','pv_uv_11','cart_ipv_11','cart_uv_11','collect_uv_11','num_gmv_11','amt_gmv_11','qty_gmv_11','unum_gmv_11','amt_alipay_11','num_alipay_11','qty_alipay_11','unum_alipay_11','ztc_pv_ipv_11','tbk_pv_ipv_11','ss_pv_ipv_11','jhs_pv_ipv_11','ztc_pv_uv_11','tbk_pv_uv_11','ss_pv_uv_11','jhs_pv_uv_11','num_alipay_njhs_11','amt_alipay_njhs_11','qty_alipay_njhs_11','unum_alipay_njhs_11','pv_ipv_sum_11','pv_uv_sum_11','cart_ipv_sum_11','cart_uv_sum_11','collect_uv_sum_11','num_gmv_sum_11','amt_gmv_sum_11','qty_gmv_sum_11','unum_gmv_sum_11','amt_alipay_sum_11','num_alipay_sum_11','qty_alipay_sum_11','unum_alipay_sum_11','ztc_pv_ipv_sum_11','tbk_pv_ipv_sum_11','ss_pv_ipv_sum_11','jhs_pv_ipv_sum_11','ztc_pv_uv_sum_11','tbk_pv_uv_sum_11','ss_pv_uv_sum_11','jhs_pv_uv_sum_11','num_alipay_njhs_sum_11','amt_alipay_njhs_sum_11','qty_alipay_njhs_sum_11','unum_alipay_njhs_sum_11','pv_ipv_14','pv_uv_14','cart_ipv_14','cart_uv_14','collect_uv_14','num_gmv_14','amt_gmv_14','qty_gmv_14','unum_gmv_14','amt_alipay_14','num_alipay_14','qty_alipay_14','unum_alipay_14','ztc_pv_ipv_14','tbk_pv_ipv_14','ss_pv_ipv_14','jhs_pv_ipv_14','ztc_pv_uv_14','tbk_pv_uv_14','ss_pv_uv_14','jhs_pv_uv_14','num_alipay_njhs_14','amt_alipay_njhs_14','qty_alipay_njhs_14','unum_alipay_njhs_14','pv_ipv_sum_14','pv_uv_sum_14','cart_ipv_sum_14','cart_uv_sum_14','collect_uv_sum_14','num_gmv_sum_14','amt_gmv_sum_14','qty_gmv_sum_14','unum_gmv_sum_14','amt_alipay_sum_14','num_alipay_sum_14','qty_alipay_sum_14','unum_alipay_sum_14','ztc_pv_ipv_sum_14','tbk_pv_ipv_sum_14','ss_pv_ipv_sum_14','jhs_pv_ipv_sum_14','ztc_pv_uv_sum_14','tbk_pv_uv_sum_14','ss_pv_uv_sum_14','jhs_pv_uv_sum_14','num_alipay_njhs_sum_14','amt_alipay_njhs_sum_14','qty_alipay_njhs_sum_14','unum_alipay_njhs_sum_14','max_qty_alipay_njhs','min_qty_alipay_njhs','std_qty_alipay_njhs','cate_id_qty_alipay_njhs_sum','cate_id_qty_alipay_njhs_avg','cate_id_qty_alipay_njhs_std','cate_level_id_qty_alipay_njhs_sum','cate_level_id_qty_alipay_njhs_avg','cate_level_id_qty_alipay_njhs_std','brand_id_qty_alipay_njhs_sum','brand_id_qty_alipay_njhs_avg','brand_id_qty_alipay_njhs_std','supplier_id_qty_alipay_njhs_sum','supplier_id_qty_alipay_njhs_avg','supplier_id_qty_alipay_njhs_std']
test_all = pd.read_csv('/Applications/dev/tmp/test_all.csv',header=None)
test_all.columns = ['watch','store_code','item_id','label','pv_ipv_1','pv_uv_1','cart_ipv_1','cart_uv_1','collect_uv_1','num_gmv_1','amt_gmv_1','qty_gmv_1','unum_gmv_1','amt_alipay_1','num_alipay_1','qty_alipay_1','unum_alipay_1','ztc_pv_ipv_1','tbk_pv_ipv_1','ss_pv_ipv_1','jhs_pv_ipv_1','ztc_pv_uv_1','tbk_pv_uv_1','ss_pv_uv_1','jhs_pv_uv_1','num_alipay_njhs_1','amt_alipay_njhs_1','qty_alipay_njhs_1','unum_alipay_njhs_1','pv_ipv_sum_1','pv_uv_sum_1','cart_ipv_sum_1','cart_uv_sum_1','collect_uv_sum_1','num_gmv_sum_1','amt_gmv_sum_1','qty_gmv_sum_1','unum_gmv_sum_1','amt_alipay_sum_1','num_alipay_sum_1','qty_alipay_sum_1','unum_alipay_sum_1','ztc_pv_ipv_sum_1','tbk_pv_ipv_sum_1','ss_pv_ipv_sum_1','jhs_pv_ipv_sum_1','ztc_pv_uv_sum_1','tbk_pv_uv_sum_1','ss_pv_uv_sum_1','jhs_pv_uv_sum_1','num_alipay_njhs_sum_1','amt_alipay_njhs_sum_1','qty_alipay_njhs_sum_1','unum_alipay_njhs_sum_1','pv_ipv_2','pv_uv_2','cart_ipv_2','cart_uv_2','collect_uv_2','num_gmv_2','amt_gmv_2','qty_gmv_2','unum_gmv_2','amt_alipay_2','num_alipay_2','qty_alipay_2','unum_alipay_2','ztc_pv_ipv_2','tbk_pv_ipv_2','ss_pv_ipv_2','jhs_pv_ipv_2','ztc_pv_uv_2','tbk_pv_uv_2','ss_pv_uv_2','jhs_pv_uv_2','num_alipay_njhs_2','amt_alipay_njhs_2','qty_alipay_njhs_2','unum_alipay_njhs_2','pv_ipv_sum_2','pv_uv_sum_2','cart_ipv_sum_2','cart_uv_sum_2','collect_uv_sum_2','num_gmv_sum_2','amt_gmv_sum_2','qty_gmv_sum_2','unum_gmv_sum_2','amt_alipay_sum_2','num_alipay_sum_2','qty_alipay_sum_2','unum_alipay_sum_2','ztc_pv_ipv_sum_2','tbk_pv_ipv_sum_2','ss_pv_ipv_sum_2','jhs_pv_ipv_sum_2','ztc_pv_uv_sum_2','tbk_pv_uv_sum_2','ss_pv_uv_sum_2','jhs_pv_uv_sum_2','num_alipay_njhs_sum_2','amt_alipay_njhs_sum_2','qty_alipay_njhs_sum_2','unum_alipay_njhs_sum_2','pv_ipv_3','pv_uv_3','cart_ipv_3','cart_uv_3','collect_uv_3','num_gmv_3','amt_gmv_3','qty_gmv_3','unum_gmv_3','amt_alipay_3','num_alipay_3','qty_alipay_3','unum_alipay_3','ztc_pv_ipv_3','tbk_pv_ipv_3','ss_pv_ipv_3','jhs_pv_ipv_3','ztc_pv_uv_3','tbk_pv_uv_3','ss_pv_uv_3','jhs_pv_uv_3','num_alipay_njhs_3','amt_alipay_njhs_3','qty_alipay_njhs_3','unum_alipay_njhs_3','pv_ipv_sum_3','pv_uv_sum_3','cart_ipv_sum_3','cart_uv_sum_3','collect_uv_sum_3','num_gmv_sum_3','amt_gmv_sum_3','qty_gmv_sum_3','unum_gmv_sum_3','amt_alipay_sum_3','num_alipay_sum_3','qty_alipay_sum_3','unum_alipay_sum_3','ztc_pv_ipv_sum_3','tbk_pv_ipv_sum_3','ss_pv_ipv_sum_3','jhs_pv_ipv_sum_3','ztc_pv_uv_sum_3','tbk_pv_uv_sum_3','ss_pv_uv_sum_3','jhs_pv_uv_sum_3','num_alipay_njhs_sum_3','amt_alipay_njhs_sum_3','qty_alipay_njhs_sum_3','unum_alipay_njhs_sum_3','pv_ipv_5','pv_uv_5','cart_ipv_5','cart_uv_5','collect_uv_5','num_gmv_5','amt_gmv_5','qty_gmv_5','unum_gmv_5','amt_alipay_5','num_alipay_5','qty_alipay_5','unum_alipay_5','ztc_pv_ipv_5','tbk_pv_ipv_5','ss_pv_ipv_5','jhs_pv_ipv_5','ztc_pv_uv_5','tbk_pv_uv_5','ss_pv_uv_5','jhs_pv_uv_5','num_alipay_njhs_5','amt_alipay_njhs_5','qty_alipay_njhs_5','unum_alipay_njhs_5','pv_ipv_sum_5','pv_uv_sum_5','cart_ipv_sum_5','cart_uv_sum_5','collect_uv_sum_5','num_gmv_sum_5','amt_gmv_sum_5','qty_gmv_sum_5','unum_gmv_sum_5','amt_alipay_sum_5','num_alipay_sum_5','qty_alipay_sum_5','unum_alipay_sum_5','ztc_pv_ipv_sum_5','tbk_pv_ipv_sum_5','ss_pv_ipv_sum_5','jhs_pv_ipv_sum_5','ztc_pv_uv_sum_5','tbk_pv_uv_sum_5','ss_pv_uv_sum_5','jhs_pv_uv_sum_5','num_alipay_njhs_sum_5','amt_alipay_njhs_sum_5','qty_alipay_njhs_sum_5','unum_alipay_njhs_sum_5','pv_ipv_7','pv_uv_7','cart_ipv_7','cart_uv_7','collect_uv_7','num_gmv_7','amt_gmv_7','qty_gmv_7','unum_gmv_7','amt_alipay_7','num_alipay_7','qty_alipay_7','unum_alipay_7','ztc_pv_ipv_7','tbk_pv_ipv_7','ss_pv_ipv_7','jhs_pv_ipv_7','ztc_pv_uv_7','tbk_pv_uv_7','ss_pv_uv_7','jhs_pv_uv_7','num_alipay_njhs_7','amt_alipay_njhs_7','qty_alipay_njhs_7','unum_alipay_njhs_7','pv_ipv_sum_7','pv_uv_sum_7','cart_ipv_sum_7','cart_uv_sum_7','collect_uv_sum_7','num_gmv_sum_7','amt_gmv_sum_7','qty_gmv_sum_7','unum_gmv_sum_7','amt_alipay_sum_7','num_alipay_sum_7','qty_alipay_sum_7','unum_alipay_sum_7','ztc_pv_ipv_sum_7','tbk_pv_ipv_sum_7','ss_pv_ipv_sum_7','jhs_pv_ipv_sum_7','ztc_pv_uv_sum_7','tbk_pv_uv_sum_7','ss_pv_uv_sum_7','jhs_pv_uv_sum_7','num_alipay_njhs_sum_7','amt_alipay_njhs_sum_7','qty_alipay_njhs_sum_7','unum_alipay_njhs_sum_7','pv_ipv_9','pv_uv_9','cart_ipv_9','cart_uv_9','collect_uv_9','num_gmv_9','amt_gmv_9','qty_gmv_9','unum_gmv_9','amt_alipay_9','num_alipay_9','qty_alipay_9','unum_alipay_9','ztc_pv_ipv_9','tbk_pv_ipv_9','ss_pv_ipv_9','jhs_pv_ipv_9','ztc_pv_uv_9','tbk_pv_uv_9','ss_pv_uv_9','jhs_pv_uv_9','num_alipay_njhs_9','amt_alipay_njhs_9','qty_alipay_njhs_9','unum_alipay_njhs_9','pv_ipv_sum_9','pv_uv_sum_9','cart_ipv_sum_9','cart_uv_sum_9','collect_uv_sum_9','num_gmv_sum_9','amt_gmv_sum_9','qty_gmv_sum_9','unum_gmv_sum_9','amt_alipay_sum_9','num_alipay_sum_9','qty_alipay_sum_9','unum_alipay_sum_9','ztc_pv_ipv_sum_9','tbk_pv_ipv_sum_9','ss_pv_ipv_sum_9','jhs_pv_ipv_sum_9','ztc_pv_uv_sum_9','tbk_pv_uv_sum_9','ss_pv_uv_sum_9','jhs_pv_uv_sum_9','num_alipay_njhs_sum_9','amt_alipay_njhs_sum_9','qty_alipay_njhs_sum_9','unum_alipay_njhs_sum_9','pv_ipv_11','pv_uv_11','cart_ipv_11','cart_uv_11','collect_uv_11','num_gmv_11','amt_gmv_11','qty_gmv_11','unum_gmv_11','amt_alipay_11','num_alipay_11','qty_alipay_11','unum_alipay_11','ztc_pv_ipv_11','tbk_pv_ipv_11','ss_pv_ipv_11','jhs_pv_ipv_11','ztc_pv_uv_11','tbk_pv_uv_11','ss_pv_uv_11','jhs_pv_uv_11','num_alipay_njhs_11','amt_alipay_njhs_11','qty_alipay_njhs_11','unum_alipay_njhs_11','pv_ipv_sum_11','pv_uv_sum_11','cart_ipv_sum_11','cart_uv_sum_11','collect_uv_sum_11','num_gmv_sum_11','amt_gmv_sum_11','qty_gmv_sum_11','unum_gmv_sum_11','amt_alipay_sum_11','num_alipay_sum_11','qty_alipay_sum_11','unum_alipay_sum_11','ztc_pv_ipv_sum_11','tbk_pv_ipv_sum_11','ss_pv_ipv_sum_11','jhs_pv_ipv_sum_11','ztc_pv_uv_sum_11','tbk_pv_uv_sum_11','ss_pv_uv_sum_11','jhs_pv_uv_sum_11','num_alipay_njhs_sum_11','amt_alipay_njhs_sum_11','qty_alipay_njhs_sum_11','unum_alipay_njhs_sum_11','pv_ipv_14','pv_uv_14','cart_ipv_14','cart_uv_14','collect_uv_14','num_gmv_14','amt_gmv_14','qty_gmv_14','unum_gmv_14','amt_alipay_14','num_alipay_14','qty_alipay_14','unum_alipay_14','ztc_pv_ipv_14','tbk_pv_ipv_14','ss_pv_ipv_14','jhs_pv_ipv_14','ztc_pv_uv_14','tbk_pv_uv_14','ss_pv_uv_14','jhs_pv_uv_14','num_alipay_njhs_14','amt_alipay_njhs_14','qty_alipay_njhs_14','unum_alipay_njhs_14','pv_ipv_sum_14','pv_uv_sum_14','cart_ipv_sum_14','cart_uv_sum_14','collect_uv_sum_14','num_gmv_sum_14','amt_gmv_sum_14','qty_gmv_sum_14','unum_gmv_sum_14','amt_alipay_sum_14','num_alipay_sum_14','qty_alipay_sum_14','unum_alipay_sum_14','ztc_pv_ipv_sum_14','tbk_pv_ipv_sum_14','ss_pv_ipv_sum_14','jhs_pv_ipv_sum_14','ztc_pv_uv_sum_14','tbk_pv_uv_sum_14','ss_pv_uv_sum_14','jhs_pv_uv_sum_14','num_alipay_njhs_sum_14','amt_alipay_njhs_sum_14','qty_alipay_njhs_sum_14','unum_alipay_njhs_sum_14','max_qty_alipay_njhs','min_qty_alipay_njhs','std_qty_alipay_njhs','cate_id_qty_alipay_njhs_sum','cate_id_qty_alipay_njhs_avg','cate_id_qty_alipay_njhs_std','cate_level_id_qty_alipay_njhs_sum','cate_level_id_qty_alipay_njhs_avg','cate_level_id_qty_alipay_njhs_std','brand_id_qty_alipay_njhs_sum','brand_id_qty_alipay_njhs_avg','brand_id_qty_alipay_njhs_std','supplier_id_qty_alipay_njhs_sum','supplier_id_qty_alipay_njhs_avg','supplier_id_qty_alipay_njhs_std']
train_fencang = pd.read_csv('/Applications/dev/tmp/train_fencang.csv',header=None)
train_fencang.columns = ['watch','store_code','item_id','label','pv_ipv_1','pv_uv_1','cart_ipv_1','cart_uv_1','collect_uv_1','num_gmv_1','amt_gmv_1','qty_gmv_1','unum_gmv_1','amt_alipay_1','num_alipay_1','qty_alipay_1','unum_alipay_1','ztc_pv_ipv_1','tbk_pv_ipv_1','ss_pv_ipv_1','jhs_pv_ipv_1','ztc_pv_uv_1','tbk_pv_uv_1','ss_pv_uv_1','jhs_pv_uv_1','num_alipay_njhs_1','amt_alipay_njhs_1','qty_alipay_njhs_1','unum_alipay_njhs_1','pv_ipv_sum_1','pv_uv_sum_1','cart_ipv_sum_1','cart_uv_sum_1','collect_uv_sum_1','num_gmv_sum_1','amt_gmv_sum_1','qty_gmv_sum_1','unum_gmv_sum_1','amt_alipay_sum_1','num_alipay_sum_1','qty_alipay_sum_1','unum_alipay_sum_1','ztc_pv_ipv_sum_1','tbk_pv_ipv_sum_1','ss_pv_ipv_sum_1','jhs_pv_ipv_sum_1','ztc_pv_uv_sum_1','tbk_pv_uv_sum_1','ss_pv_uv_sum_1','jhs_pv_uv_sum_1','num_alipay_njhs_sum_1','amt_alipay_njhs_sum_1','qty_alipay_njhs_sum_1','unum_alipay_njhs_sum_1','pv_ipv_2','pv_uv_2','cart_ipv_2','cart_uv_2','collect_uv_2','num_gmv_2','amt_gmv_2','qty_gmv_2','unum_gmv_2','amt_alipay_2','num_alipay_2','qty_alipay_2','unum_alipay_2','ztc_pv_ipv_2','tbk_pv_ipv_2','ss_pv_ipv_2','jhs_pv_ipv_2','ztc_pv_uv_2','tbk_pv_uv_2','ss_pv_uv_2','jhs_pv_uv_2','num_alipay_njhs_2','amt_alipay_njhs_2','qty_alipay_njhs_2','unum_alipay_njhs_2','pv_ipv_sum_2','pv_uv_sum_2','cart_ipv_sum_2','cart_uv_sum_2','collect_uv_sum_2','num_gmv_sum_2','amt_gmv_sum_2','qty_gmv_sum_2','unum_gmv_sum_2','amt_alipay_sum_2','num_alipay_sum_2','qty_alipay_sum_2','unum_alipay_sum_2','ztc_pv_ipv_sum_2','tbk_pv_ipv_sum_2','ss_pv_ipv_sum_2','jhs_pv_ipv_sum_2','ztc_pv_uv_sum_2','tbk_pv_uv_sum_2','ss_pv_uv_sum_2','jhs_pv_uv_sum_2','num_alipay_njhs_sum_2','amt_alipay_njhs_sum_2','qty_alipay_njhs_sum_2','unum_alipay_njhs_sum_2','pv_ipv_3','pv_uv_3','cart_ipv_3','cart_uv_3','collect_uv_3','num_gmv_3','amt_gmv_3','qty_gmv_3','unum_gmv_3','amt_alipay_3','num_alipay_3','qty_alipay_3','unum_alipay_3','ztc_pv_ipv_3','tbk_pv_ipv_3','ss_pv_ipv_3','jhs_pv_ipv_3','ztc_pv_uv_3','tbk_pv_uv_3','ss_pv_uv_3','jhs_pv_uv_3','num_alipay_njhs_3','amt_alipay_njhs_3','qty_alipay_njhs_3','unum_alipay_njhs_3','pv_ipv_sum_3','pv_uv_sum_3','cart_ipv_sum_3','cart_uv_sum_3','collect_uv_sum_3','num_gmv_sum_3','amt_gmv_sum_3','qty_gmv_sum_3','unum_gmv_sum_3','amt_alipay_sum_3','num_alipay_sum_3','qty_alipay_sum_3','unum_alipay_sum_3','ztc_pv_ipv_sum_3','tbk_pv_ipv_sum_3','ss_pv_ipv_sum_3','jhs_pv_ipv_sum_3','ztc_pv_uv_sum_3','tbk_pv_uv_sum_3','ss_pv_uv_sum_3','jhs_pv_uv_sum_3','num_alipay_njhs_sum_3','amt_alipay_njhs_sum_3','qty_alipay_njhs_sum_3','unum_alipay_njhs_sum_3','pv_ipv_5','pv_uv_5','cart_ipv_5','cart_uv_5','collect_uv_5','num_gmv_5','amt_gmv_5','qty_gmv_5','unum_gmv_5','amt_alipay_5','num_alipay_5','qty_alipay_5','unum_alipay_5','ztc_pv_ipv_5','tbk_pv_ipv_5','ss_pv_ipv_5','jhs_pv_ipv_5','ztc_pv_uv_5','tbk_pv_uv_5','ss_pv_uv_5','jhs_pv_uv_5','num_alipay_njhs_5','amt_alipay_njhs_5','qty_alipay_njhs_5','unum_alipay_njhs_5','pv_ipv_sum_5','pv_uv_sum_5','cart_ipv_sum_5','cart_uv_sum_5','collect_uv_sum_5','num_gmv_sum_5','amt_gmv_sum_5','qty_gmv_sum_5','unum_gmv_sum_5','amt_alipay_sum_5','num_alipay_sum_5','qty_alipay_sum_5','unum_alipay_sum_5','ztc_pv_ipv_sum_5','tbk_pv_ipv_sum_5','ss_pv_ipv_sum_5','jhs_pv_ipv_sum_5','ztc_pv_uv_sum_5','tbk_pv_uv_sum_5','ss_pv_uv_sum_5','jhs_pv_uv_sum_5','num_alipay_njhs_sum_5','amt_alipay_njhs_sum_5','qty_alipay_njhs_sum_5','unum_alipay_njhs_sum_5','pv_ipv_7','pv_uv_7','cart_ipv_7','cart_uv_7','collect_uv_7','num_gmv_7','amt_gmv_7','qty_gmv_7','unum_gmv_7','amt_alipay_7','num_alipay_7','qty_alipay_7','unum_alipay_7','ztc_pv_ipv_7','tbk_pv_ipv_7','ss_pv_ipv_7','jhs_pv_ipv_7','ztc_pv_uv_7','tbk_pv_uv_7','ss_pv_uv_7','jhs_pv_uv_7','num_alipay_njhs_7','amt_alipay_njhs_7','qty_alipay_njhs_7','unum_alipay_njhs_7','pv_ipv_sum_7','pv_uv_sum_7','cart_ipv_sum_7','cart_uv_sum_7','collect_uv_sum_7','num_gmv_sum_7','amt_gmv_sum_7','qty_gmv_sum_7','unum_gmv_sum_7','amt_alipay_sum_7','num_alipay_sum_7','qty_alipay_sum_7','unum_alipay_sum_7','ztc_pv_ipv_sum_7','tbk_pv_ipv_sum_7','ss_pv_ipv_sum_7','jhs_pv_ipv_sum_7','ztc_pv_uv_sum_7','tbk_pv_uv_sum_7','ss_pv_uv_sum_7','jhs_pv_uv_sum_7','num_alipay_njhs_sum_7','amt_alipay_njhs_sum_7','qty_alipay_njhs_sum_7','unum_alipay_njhs_sum_7','pv_ipv_9','pv_uv_9','cart_ipv_9','cart_uv_9','collect_uv_9','num_gmv_9','amt_gmv_9','qty_gmv_9','unum_gmv_9','amt_alipay_9','num_alipay_9','qty_alipay_9','unum_alipay_9','ztc_pv_ipv_9','tbk_pv_ipv_9','ss_pv_ipv_9','jhs_pv_ipv_9','ztc_pv_uv_9','tbk_pv_uv_9','ss_pv_uv_9','jhs_pv_uv_9','num_alipay_njhs_9','amt_alipay_njhs_9','qty_alipay_njhs_9','unum_alipay_njhs_9','pv_ipv_sum_9','pv_uv_sum_9','cart_ipv_sum_9','cart_uv_sum_9','collect_uv_sum_9','num_gmv_sum_9','amt_gmv_sum_9','qty_gmv_sum_9','unum_gmv_sum_9','amt_alipay_sum_9','num_alipay_sum_9','qty_alipay_sum_9','unum_alipay_sum_9','ztc_pv_ipv_sum_9','tbk_pv_ipv_sum_9','ss_pv_ipv_sum_9','jhs_pv_ipv_sum_9','ztc_pv_uv_sum_9','tbk_pv_uv_sum_9','ss_pv_uv_sum_9','jhs_pv_uv_sum_9','num_alipay_njhs_sum_9','amt_alipay_njhs_sum_9','qty_alipay_njhs_sum_9','unum_alipay_njhs_sum_9','pv_ipv_11','pv_uv_11','cart_ipv_11','cart_uv_11','collect_uv_11','num_gmv_11','amt_gmv_11','qty_gmv_11','unum_gmv_11','amt_alipay_11','num_alipay_11','qty_alipay_11','unum_alipay_11','ztc_pv_ipv_11','tbk_pv_ipv_11','ss_pv_ipv_11','jhs_pv_ipv_11','ztc_pv_uv_11','tbk_pv_uv_11','ss_pv_uv_11','jhs_pv_uv_11','num_alipay_njhs_11','amt_alipay_njhs_11','qty_alipay_njhs_11','unum_alipay_njhs_11','pv_ipv_sum_11','pv_uv_sum_11','cart_ipv_sum_11','cart_uv_sum_11','collect_uv_sum_11','num_gmv_sum_11','amt_gmv_sum_11','qty_gmv_sum_11','unum_gmv_sum_11','amt_alipay_sum_11','num_alipay_sum_11','qty_alipay_sum_11','unum_alipay_sum_11','ztc_pv_ipv_sum_11','tbk_pv_ipv_sum_11','ss_pv_ipv_sum_11','jhs_pv_ipv_sum_11','ztc_pv_uv_sum_11','tbk_pv_uv_sum_11','ss_pv_uv_sum_11','jhs_pv_uv_sum_11','num_alipay_njhs_sum_11','amt_alipay_njhs_sum_11','qty_alipay_njhs_sum_11','unum_alipay_njhs_sum_11','pv_ipv_14','pv_uv_14','cart_ipv_14','cart_uv_14','collect_uv_14','num_gmv_14','amt_gmv_14','qty_gmv_14','unum_gmv_14','amt_alipay_14','num_alipay_14','qty_alipay_14','unum_alipay_14','ztc_pv_ipv_14','tbk_pv_ipv_14','ss_pv_ipv_14','jhs_pv_ipv_14','ztc_pv_uv_14','tbk_pv_uv_14','ss_pv_uv_14','jhs_pv_uv_14','num_alipay_njhs_14','amt_alipay_njhs_14','qty_alipay_njhs_14','unum_alipay_njhs_14','pv_ipv_sum_14','pv_uv_sum_14','cart_ipv_sum_14','cart_uv_sum_14','collect_uv_sum_14','num_gmv_sum_14','amt_gmv_sum_14','qty_gmv_sum_14','unum_gmv_sum_14','amt_alipay_sum_14','num_alipay_sum_14','qty_alipay_sum_14','unum_alipay_sum_14','ztc_pv_ipv_sum_14','tbk_pv_ipv_sum_14','ss_pv_ipv_sum_14','jhs_pv_ipv_sum_14','ztc_pv_uv_sum_14','tbk_pv_uv_sum_14','ss_pv_uv_sum_14','jhs_pv_uv_sum_14','num_alipay_njhs_sum_14','amt_alipay_njhs_sum_14','qty_alipay_njhs_sum_14','unum_alipay_njhs_sum_14','max_qty_alipay_njhs','min_qty_alipay_njhs','std_qty_alipay_njhs','cate_id_qty_alipay_njhs_sum','cate_id_qty_alipay_njhs_avg','cate_id_qty_alipay_njhs_std','cate_level_id_qty_alipay_njhs_sum','cate_level_id_qty_alipay_njhs_avg','cate_level_id_qty_alipay_njhs_std','brand_id_qty_alipay_njhs_sum','brand_id_qty_alipay_njhs_avg','brand_id_qty_alipay_njhs_std','supplier_id_qty_alipay_njhs_sum','supplier_id_qty_alipay_njhs_avg','supplier_id_qty_alipay_njhs_std']
test_fencang = pd.read_csv('/Applications/dev/tmp/test_fencang.csv',header=None)
test_fencang.columns = ['watch','store_code','item_id','label','pv_ipv_1','pv_uv_1','cart_ipv_1','cart_uv_1','collect_uv_1','num_gmv_1','amt_gmv_1','qty_gmv_1','unum_gmv_1','amt_alipay_1','num_alipay_1','qty_alipay_1','unum_alipay_1','ztc_pv_ipv_1','tbk_pv_ipv_1','ss_pv_ipv_1','jhs_pv_ipv_1','ztc_pv_uv_1','tbk_pv_uv_1','ss_pv_uv_1','jhs_pv_uv_1','num_alipay_njhs_1','amt_alipay_njhs_1','qty_alipay_njhs_1','unum_alipay_njhs_1','pv_ipv_sum_1','pv_uv_sum_1','cart_ipv_sum_1','cart_uv_sum_1','collect_uv_sum_1','num_gmv_sum_1','amt_gmv_sum_1','qty_gmv_sum_1','unum_gmv_sum_1','amt_alipay_sum_1','num_alipay_sum_1','qty_alipay_sum_1','unum_alipay_sum_1','ztc_pv_ipv_sum_1','tbk_pv_ipv_sum_1','ss_pv_ipv_sum_1','jhs_pv_ipv_sum_1','ztc_pv_uv_sum_1','tbk_pv_uv_sum_1','ss_pv_uv_sum_1','jhs_pv_uv_sum_1','num_alipay_njhs_sum_1','amt_alipay_njhs_sum_1','qty_alipay_njhs_sum_1','unum_alipay_njhs_sum_1','pv_ipv_2','pv_uv_2','cart_ipv_2','cart_uv_2','collect_uv_2','num_gmv_2','amt_gmv_2','qty_gmv_2','unum_gmv_2','amt_alipay_2','num_alipay_2','qty_alipay_2','unum_alipay_2','ztc_pv_ipv_2','tbk_pv_ipv_2','ss_pv_ipv_2','jhs_pv_ipv_2','ztc_pv_uv_2','tbk_pv_uv_2','ss_pv_uv_2','jhs_pv_uv_2','num_alipay_njhs_2','amt_alipay_njhs_2','qty_alipay_njhs_2','unum_alipay_njhs_2','pv_ipv_sum_2','pv_uv_sum_2','cart_ipv_sum_2','cart_uv_sum_2','collect_uv_sum_2','num_gmv_sum_2','amt_gmv_sum_2','qty_gmv_sum_2','unum_gmv_sum_2','amt_alipay_sum_2','num_alipay_sum_2','qty_alipay_sum_2','unum_alipay_sum_2','ztc_pv_ipv_sum_2','tbk_pv_ipv_sum_2','ss_pv_ipv_sum_2','jhs_pv_ipv_sum_2','ztc_pv_uv_sum_2','tbk_pv_uv_sum_2','ss_pv_uv_sum_2','jhs_pv_uv_sum_2','num_alipay_njhs_sum_2','amt_alipay_njhs_sum_2','qty_alipay_njhs_sum_2','unum_alipay_njhs_sum_2','pv_ipv_3','pv_uv_3','cart_ipv_3','cart_uv_3','collect_uv_3','num_gmv_3','amt_gmv_3','qty_gmv_3','unum_gmv_3','amt_alipay_3','num_alipay_3','qty_alipay_3','unum_alipay_3','ztc_pv_ipv_3','tbk_pv_ipv_3','ss_pv_ipv_3','jhs_pv_ipv_3','ztc_pv_uv_3','tbk_pv_uv_3','ss_pv_uv_3','jhs_pv_uv_3','num_alipay_njhs_3','amt_alipay_njhs_3','qty_alipay_njhs_3','unum_alipay_njhs_3','pv_ipv_sum_3','pv_uv_sum_3','cart_ipv_sum_3','cart_uv_sum_3','collect_uv_sum_3','num_gmv_sum_3','amt_gmv_sum_3','qty_gmv_sum_3','unum_gmv_sum_3','amt_alipay_sum_3','num_alipay_sum_3','qty_alipay_sum_3','unum_alipay_sum_3','ztc_pv_ipv_sum_3','tbk_pv_ipv_sum_3','ss_pv_ipv_sum_3','jhs_pv_ipv_sum_3','ztc_pv_uv_sum_3','tbk_pv_uv_sum_3','ss_pv_uv_sum_3','jhs_pv_uv_sum_3','num_alipay_njhs_sum_3','amt_alipay_njhs_sum_3','qty_alipay_njhs_sum_3','unum_alipay_njhs_sum_3','pv_ipv_5','pv_uv_5','cart_ipv_5','cart_uv_5','collect_uv_5','num_gmv_5','amt_gmv_5','qty_gmv_5','unum_gmv_5','amt_alipay_5','num_alipay_5','qty_alipay_5','unum_alipay_5','ztc_pv_ipv_5','tbk_pv_ipv_5','ss_pv_ipv_5','jhs_pv_ipv_5','ztc_pv_uv_5','tbk_pv_uv_5','ss_pv_uv_5','jhs_pv_uv_5','num_alipay_njhs_5','amt_alipay_njhs_5','qty_alipay_njhs_5','unum_alipay_njhs_5','pv_ipv_sum_5','pv_uv_sum_5','cart_ipv_sum_5','cart_uv_sum_5','collect_uv_sum_5','num_gmv_sum_5','amt_gmv_sum_5','qty_gmv_sum_5','unum_gmv_sum_5','amt_alipay_sum_5','num_alipay_sum_5','qty_alipay_sum_5','unum_alipay_sum_5','ztc_pv_ipv_sum_5','tbk_pv_ipv_sum_5','ss_pv_ipv_sum_5','jhs_pv_ipv_sum_5','ztc_pv_uv_sum_5','tbk_pv_uv_sum_5','ss_pv_uv_sum_5','jhs_pv_uv_sum_5','num_alipay_njhs_sum_5','amt_alipay_njhs_sum_5','qty_alipay_njhs_sum_5','unum_alipay_njhs_sum_5','pv_ipv_7','pv_uv_7','cart_ipv_7','cart_uv_7','collect_uv_7','num_gmv_7','amt_gmv_7','qty_gmv_7','unum_gmv_7','amt_alipay_7','num_alipay_7','qty_alipay_7','unum_alipay_7','ztc_pv_ipv_7','tbk_pv_ipv_7','ss_pv_ipv_7','jhs_pv_ipv_7','ztc_pv_uv_7','tbk_pv_uv_7','ss_pv_uv_7','jhs_pv_uv_7','num_alipay_njhs_7','amt_alipay_njhs_7','qty_alipay_njhs_7','unum_alipay_njhs_7','pv_ipv_sum_7','pv_uv_sum_7','cart_ipv_sum_7','cart_uv_sum_7','collect_uv_sum_7','num_gmv_sum_7','amt_gmv_sum_7','qty_gmv_sum_7','unum_gmv_sum_7','amt_alipay_sum_7','num_alipay_sum_7','qty_alipay_sum_7','unum_alipay_sum_7','ztc_pv_ipv_sum_7','tbk_pv_ipv_sum_7','ss_pv_ipv_sum_7','jhs_pv_ipv_sum_7','ztc_pv_uv_sum_7','tbk_pv_uv_sum_7','ss_pv_uv_sum_7','jhs_pv_uv_sum_7','num_alipay_njhs_sum_7','amt_alipay_njhs_sum_7','qty_alipay_njhs_sum_7','unum_alipay_njhs_sum_7','pv_ipv_9','pv_uv_9','cart_ipv_9','cart_uv_9','collect_uv_9','num_gmv_9','amt_gmv_9','qty_gmv_9','unum_gmv_9','amt_alipay_9','num_alipay_9','qty_alipay_9','unum_alipay_9','ztc_pv_ipv_9','tbk_pv_ipv_9','ss_pv_ipv_9','jhs_pv_ipv_9','ztc_pv_uv_9','tbk_pv_uv_9','ss_pv_uv_9','jhs_pv_uv_9','num_alipay_njhs_9','amt_alipay_njhs_9','qty_alipay_njhs_9','unum_alipay_njhs_9','pv_ipv_sum_9','pv_uv_sum_9','cart_ipv_sum_9','cart_uv_sum_9','collect_uv_sum_9','num_gmv_sum_9','amt_gmv_sum_9','qty_gmv_sum_9','unum_gmv_sum_9','amt_alipay_sum_9','num_alipay_sum_9','qty_alipay_sum_9','unum_alipay_sum_9','ztc_pv_ipv_sum_9','tbk_pv_ipv_sum_9','ss_pv_ipv_sum_9','jhs_pv_ipv_sum_9','ztc_pv_uv_sum_9','tbk_pv_uv_sum_9','ss_pv_uv_sum_9','jhs_pv_uv_sum_9','num_alipay_njhs_sum_9','amt_alipay_njhs_sum_9','qty_alipay_njhs_sum_9','unum_alipay_njhs_sum_9','pv_ipv_11','pv_uv_11','cart_ipv_11','cart_uv_11','collect_uv_11','num_gmv_11','amt_gmv_11','qty_gmv_11','unum_gmv_11','amt_alipay_11','num_alipay_11','qty_alipay_11','unum_alipay_11','ztc_pv_ipv_11','tbk_pv_ipv_11','ss_pv_ipv_11','jhs_pv_ipv_11','ztc_pv_uv_11','tbk_pv_uv_11','ss_pv_uv_11','jhs_pv_uv_11','num_alipay_njhs_11','amt_alipay_njhs_11','qty_alipay_njhs_11','unum_alipay_njhs_11','pv_ipv_sum_11','pv_uv_sum_11','cart_ipv_sum_11','cart_uv_sum_11','collect_uv_sum_11','num_gmv_sum_11','amt_gmv_sum_11','qty_gmv_sum_11','unum_gmv_sum_11','amt_alipay_sum_11','num_alipay_sum_11','qty_alipay_sum_11','unum_alipay_sum_11','ztc_pv_ipv_sum_11','tbk_pv_ipv_sum_11','ss_pv_ipv_sum_11','jhs_pv_ipv_sum_11','ztc_pv_uv_sum_11','tbk_pv_uv_sum_11','ss_pv_uv_sum_11','jhs_pv_uv_sum_11','num_alipay_njhs_sum_11','amt_alipay_njhs_sum_11','qty_alipay_njhs_sum_11','unum_alipay_njhs_sum_11','pv_ipv_14','pv_uv_14','cart_ipv_14','cart_uv_14','collect_uv_14','num_gmv_14','amt_gmv_14','qty_gmv_14','unum_gmv_14','amt_alipay_14','num_alipay_14','qty_alipay_14','unum_alipay_14','ztc_pv_ipv_14','tbk_pv_ipv_14','ss_pv_ipv_14','jhs_pv_ipv_14','ztc_pv_uv_14','tbk_pv_uv_14','ss_pv_uv_14','jhs_pv_uv_14','num_alipay_njhs_14','amt_alipay_njhs_14','qty_alipay_njhs_14','unum_alipay_njhs_14','pv_ipv_sum_14','pv_uv_sum_14','cart_ipv_sum_14','cart_uv_sum_14','collect_uv_sum_14','num_gmv_sum_14','amt_gmv_sum_14','qty_gmv_sum_14','unum_gmv_sum_14','amt_alipay_sum_14','num_alipay_sum_14','qty_alipay_sum_14','unum_alipay_sum_14','ztc_pv_ipv_sum_14','tbk_pv_ipv_sum_14','ss_pv_ipv_sum_14','jhs_pv_ipv_sum_14','ztc_pv_uv_sum_14','tbk_pv_uv_sum_14','ss_pv_uv_sum_14','jhs_pv_uv_sum_14','num_alipay_njhs_sum_14','amt_alipay_njhs_sum_14','qty_alipay_njhs_sum_14','unum_alipay_njhs_sum_14','max_qty_alipay_njhs','min_qty_alipay_njhs','std_qty_alipay_njhs','cate_id_qty_alipay_njhs_sum','cate_id_qty_alipay_njhs_avg','cate_id_qty_alipay_njhs_std','cate_level_id_qty_alipay_njhs_sum','cate_level_id_qty_alipay_njhs_avg','cate_level_id_qty_alipay_njhs_std','brand_id_qty_alipay_njhs_sum','brand_id_qty_alipay_njhs_avg','brand_id_qty_alipay_njhs_std','supplier_id_qty_alipay_njhs_sum','supplier_id_qty_alipay_njhs_avg','supplier_id_qty_alipay_njhs_std']


# In[283]:

train_fencang.head(5)


# In[284]:

train_all = train_all.replace("\N",0.0)
test_all = test_all.replace("\N",0.0)
train_fencang = train_fencang.replace("\N",0.0)
test_fencang = test_fencang.replace("\N",0.0)


# In[285]:

train_all = train_all.apply(lambda x: pd.to_numeric(x, errors='ignore'))
test_all = test_all.apply(lambda x: pd.to_numeric(x, errors='ignore'))
train_fencang = train_fencang.apply(lambda x: pd.to_numeric(x, errors='ignore'))
test_fencang = test_fencang.apply(lambda x: pd.to_numeric(x, errors='ignore'))


# In[414]:

train_all_fencang = pd.concat([train_all,train_fencang])


# In[415]:

train_all_fencang.store_code = train_all_fencang.store_code.apply(lambda x: str(x))


# In[416]:

train_all_fencang.store_code.dtype


# In[324]:

train_all.to_csv('../data/train_all.csv',index=None)
test_all.to_csv('../data/test_all.csv',index=None)
train_fencang.to_csv('../data/train_fencang.csv',index=None)
test_fencang.to_csv('../data/test_fencang.csv',index=None)


# In[325]:

#计算代价
def cal_cost(y,pred,a,b):
    cost = 0
    nb_sample = len(y)
    for i in range(nb_sample):
        if pred[i]>y[i]:
            cost += b[i]*(pred[i]-y[i])
        else:
            cost += a[i]*(y[i]-pred[i])
    return nb_sample,cost


# In[417]:

a_b = pd.read_csv('../data/config2.csv')


# In[418]:

train_all_fencang.shape


# In[419]:

data_train = train_all_fencang


# In[420]:

data_train.head(5)


# In[421]:

hashmap = {'1':'1','2':'2','3':'3','4':'4','5':'5','all':'6'}


# In[422]:

data_train.store_code.dtype


# In[423]:

data_train.store_code = data_train.store_code.apply(lambda x:hashmap[x])


# In[424]:

a_b.store_code = a_b.store_code.apply(lambda x:hashmap[x])


# In[426]:

data_train.shape


# In[427]:

data_train = pd.merge(data_train,a_b,on=['item_id','store_code'])


# In[460]:

def model_gbrt(data_train,store_code,loss='lad',learning_rate=0.01,n_estimators=400,subsample=0.75,max_depth=6,random_state=1024, max_features=0.75):
    starttime = datetime.datetime.now() 
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
    model = GradientBoostingRegressor(loss=loss,learning_rate=learning_rate,n_estimators=n_estimators,subsample=subsample,max_depth=max_depth,random_state=random_state, max_features=max_features)
    model.fit(train_x,train_y, sample_weight=train_weight)
    val_a_b['pred'] = model.predict(val_x)
    val_a_b['y'] = val_y
    cost = cal_cost(val_y.values,val_a_b.pred.values,val_a_b.a.values,val_a_b.b.values)
    val_a_b.to_csv('val_{0}.csv'.format(store_code),index=None)
    print "used time: " + str((endtime - starttime).seconds)    + " s" 


# In[432]:

if __name__ == "__main__":
    model_gbrt(data,'4')


# In[463]:

def model_rf(data_train,store_code,num_boost_round=400):
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
            'max_depth':7,  # 3-10
            'lambda':100,  # 控制XGBoost的正则化部分
            'subsample':0.7,  # 0.5-1
            'colsample_bytree':0.7,  # 0.5-1
            'eta': 0.09,  # 0.01-0.2
            'seed':1024,
            'nthread':8
            #'gamma':1 # 这个参数的值越大，算法越保守。这个参数的值和损失函数息息相关，所以是需要调整的
        }
    num_boost_round = num_boost_round
    watchlist  = [(dtrain,'train')]
    model = xgb.train(params,dtrain,num_boost_round=num_boost_round,evals=watchlist)        
    #predict val set
    val_a_b['pred'] = model.predict(dval)
    val_a_b['y'] = val_y
    cost = cal_cost(val_y.values,val_a_b.pred.values,val_a_b.a.values,val_a_b.b.values)
    val_a_b.to_csv('val_xgb_{0}.csv'.format(store_code),index=None)
    print "used time: " + str((endtime - starttime).seconds)    + " s" 


# In[471]:

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


# In[466]:

model_xgb(data_train,'4')
model_xgb(data_train,'5')
model_xgb(data_train,'6')


# In[476]:

if __name__ == "__main__":
    model_rf(data_train,'1',max_depth=7)
    model_rf(data_train,'2',max_depth=5)
    model_rf(data_train,'3',max_depth=7,max_features=0.7)
    model_rf(data_train,'4',max_depth=4,max_features=0.6)
    model_rf(data_train,'5',max_depth=5,max_features=0.6)
    model_rf(data_train,'6')


# In[480]:

import pandas as pd


xgb1 = pd.read_csv('../data/val_xgb_1.csv')
xgb2 = pd.read_csv('../data/val_xgb_2.csv')
xgb3 = pd.read_csv('../data/val_xgb_3.csv')
xgb4 = pd.read_csv('../data/val_xgb_4.csv')
xgb5 = pd.read_csv('../data/val_xgb_5.csv')
xgball = pd.read_csv('../data/val_xgb_6.csv')

pred = pd.concat([xgball,xgb1,xgb2,xgb3,xgb4,xgb5])
pred[['item_id','store_code','pred']].to_csv('../data/xgb.csv',index=None,header=None)


rf1 = pd.read_csv('../data/val_rf_1.csv')
rf2 = pd.read_csv('../data/val_rf_2.csv')
rf3 = pd.read_csv('../data/val_rf_3.csv')
rf4 = pd.read_csv('../data/val_rf_4.csv')
rf5 = pd.read_csv('../data/val_rf_5.csv')
rfall = pd.read_csv('../data/val_rf_6.csv')

pred = pd.concat([rfall,rf1,rf2,rf3,rf4,rf5])
pred[['item_id','store_code','pred']].to_csv('../data/rf.csv',index=None,header=None)



gbrt1 = pd.read_csv('../data/val_1.csv')
gbrt2 = pd.read_csv('../data/val_2.csv')
gbrt3 = pd.read_csv('../data/val_3.csv')
gbrt4 = pd.read_csv('../data/val_4.csv')
gbrt5 = pd.read_csv('../data/val_5.csv')
gbrtall = pd.read_csv('../data/val_6.csv')

pred = pd.concat([gbrtall,gbrt1,gbrt2,gbrt3,gbrt4,gbrt5])
pred[['item_id','store_code','pred']].to_csv('../data/gbrt.csv',index=None,header=None)


# In[498]:

item_store_feature = pd.read_csv('/Users/hyy/Downloads/CAINIAO/item_store_feature2.csv',header=None)
item_store_feature.columns = ['date','item_id','store_code','cate_id','cate_level_id','brand_id','supplier_id','pv_ipv','pv_uv','cart_ipv','cart_uv','collect_uv','num_gmv',
'amt_gmv','qty_gmv','unum_gmv','amt_alipay','num_alipay','qty_alipay','unum_alipay','ztc_pv_ipv','tbk_pv_ipv','ss_pv_ipv','jhs_pv_ipv',
'ztc_pv_uv','tbk_pv_uv','ss_pv_uv','jhs_pv_uv','num_alipay_njhs','amt_alipay_njhs','qty_alipay_njhs','unum_alipay_njhs']
item_store_feature.to_csv('../data/item_store_feature2.csv',index=None)


# In[499]:

item_feature = pd.read_csv('/Users/hyy/Downloads/CAINIAO/item_feature2.csv',header=None)
item_feature.columns = ['date','item_id','cate_id','cate_level_id','brand_id','supplier_id','pv_ipv','pv_uv','cart_ipv','cart_uv','collect_uv','num_gmv',
'amt_gmv','qty_gmv','unum_gmv','amt_alipay','num_alipay','qty_alipay','unum_alipay','ztc_pv_ipv','tbk_pv_ipv','ss_pv_ipv','jhs_pv_ipv','ztc_pv_uv',
'tbk_pv_uv','ss_pv_uv','jhs_pv_uv','num_alipay_njhs','amt_alipay_njhs','qty_alipay_njhs','unum_alipay_njhs']
item_feature.to_csv('../data/item_feature2.csv',index=None)


# In[500]:

item_all = pd.read_csv('../data/item_feature2.csv')
item_all['store_code'] = ['all' for _ in range(len(item_all))]
item_store = pd.read_csv('../data/item_store_feature2.csv')


# In[501]:

item_all.head(5)


# In[502]:

item_store.head(5)


# In[503]:

items = pd.concat([item_all,item_store])


# In[505]:

items = items[(items.date!=20151111)&(items.date!=20151212)]

items_to_pred = items[['item_id','store_code','qty_alipay_njhs']].groupby(['item_id','store_code']).agg('sum')
items_to_pred.reset_index(inplace=True)
items_to_pred.drop('qty_alipay_njhs',axis=1,inplace=True)
items_to_pred.store_code = items_to_pred.store_code.apply(lambda x:str(x))


# In[ ]:




# In[487]:

item_all = pd.read_csv('../data/item_feature2.csv')
item_all['store_code'] = ['all' for _ in range(len(item_all))]
item_store = pd.read_csv('../data/item_store_feature2.csv')

#剔除双十一双十二
items = pd.concat([item_all,item_store])
items = items[(items.date!=20151111)&(items.date!=20151212)]

items_to_pred = items[['item_id','store_code','qty_alipay_njhs']].groupby(['item_id','store_code']).agg('sum')
items_to_pred.reset_index(inplace=True)
items_to_pred.drop('qty_alipay_njhs',axis=1,inplace=True)
items_to_pred.store_code = items_to_pred.store_code.apply(lambda x:str(x))


# In[512]:

items.date = items.date.apply(lambda x:transform_date(x))


# In[515]:

# 转换日期格式
def transform_date(x):
    end = date(2015,12,28)
    if x>20151212:
        return (end-date(x/10000,x%10000/100,x%100)).days
    elif 20151111<x<20151212:
        return (end-date(x/10000,x%10000/100,x%100)).days - 1
    else:
        return (end-date(x/10000,x%10000/100,x%100)).days - 2

items.date = items.date.apply(transform_date)
items.date = items.date.apply(lambda x:(1+(x-1)/7))


# In[516]:

items = items[['item_id','store_code','date','qty_alipay_njhs']].groupby(['item_id','store_code','date']).agg('sum')
items.reset_index(inplace=True)  
items.rename(columns={'date':'week_reverse'},inplace=True)


# In[518]:

#第一二周为真实label,共5763个item store组合
items_truth = items[items.week_reverse<=2]
items_truth = items_truth[['item_id','store_code','qty_alipay_njhs']].groupby(['item_id','store_code']).agg('sum')
items_truth.reset_index(inplace=True)
items_truth.store_code = items_truth.store_code.apply(lambda x:str(x))
items_truth = pd.merge(items_to_pred,items_truth,on=['item_id','store_code'],how='left')
items_truth.fillna(0,inplace=True)
a_b = pd.read_csv('../data/config2.csv')
items_truth = pd.merge(items_truth,a_b,on=['item_id','store_code'])

items_truth[['item_id','store_code','qty_alipay_njhs']].to_csv('val_y.csv',index=None)


# In[519]:

#计算前2周的最小组合min，最大组合max，如果a>b则预测为max*2，反之min*2
items.store_code = items.store_code.apply(lambda x:str(x))
a_b = pd.read_csv('../data/config2.csv')
items = pd.merge(items,a_b,on=['item_id','store_code'])

items = items[(3<=items.week_reverse)&(items.week_reverse<=5)]


# In[520]:

items.sort(columns=['item_id','store_code','qty_alipay_njhs'],inplace=True)
t0 = items[['item_id','store_code','a','b','qty_alipay_njhs']].groupby(['item_id','store_code','a','b']).agg('nth',0)
t1 = items[['item_id','store_code','a','b','qty_alipay_njhs']].groupby(['item_id','store_code','a','b']).agg('nth',0)
t2 = items[['item_id','store_code','a','b','qty_alipay_njhs']].groupby(['item_id','store_code','a','b']).agg('nth',-1)
t3 = items[['item_id','store_code','a','b','qty_alipay_njhs']].groupby(['item_id','store_code','a','b']).agg('nth',-1)
t0.rename(columns={'qty_alipay_njhs':'min1'},inplace=True)
t1.rename(columns={'qty_alipay_njhs':'min2'},inplace=True)
t2.rename(columns={'qty_alipay_njhs':'max1'},inplace=True)
t3.rename(columns={'qty_alipay_njhs':'max2'},inplace=True)
t0.reset_index(inplace=True)
t1.reset_index(inplace=True)
t2.reset_index(inplace=True)
t3.reset_index(inplace=True)

items = pd.merge(t0,t1,on=['item_id','store_code','a','b'])
items = pd.merge(items,t2,on=['item_id','store_code','a','b'])
items = pd.merge(items,t3,on=['item_id','store_code','a','b'])

items['min_'] = items.min1+items.min2
items['max_'] = items.max1+items.max2


# In[521]:


items['pred'] = items.min_ + (items.max_ - items.min_)*(items.a**50/(items.a**50+items.b**50))
items_pred = items[['item_id','store_code','pred']]
items_pred.pred = items_pred.pred.apply(lambda x:0.0 if x<0.01 else x)


items_truth_pred = pd.merge(items_truth,items_pred,on=['item_id','store_code'],how='left')
items_truth_pred.fillna(0.0,inplace=True)


#计算代价
def cal_cost(y,pred,a,b):
    cost = 0
    nb_sample = len(y)
    for i in range(nb_sample):
        if pred[i]>y[i]:
            cost += b[i]*(pred[i]-y[i])
        else:
            cost += a[i]*(y[i]-pred[i])
    return nb_sample,cost

cost = cal_cost(items_truth_pred.qty_alipay_njhs,items_truth_pred.pred,items_truth_pred.a,items_truth_pred.b)

items_truth_pred[['item_id','store_code','pred']].to_csv('rule_{0}.csv'.format(cost[1]),index=None,header=None)


# In[522]:

def cal_cost(y,pred,a,b):
    y,pred,a,b = y.values,pred.values,a.values,b.values
    cost = 0
    nb_sample = len(y)
    for i in range(nb_sample):
        if pred[i]>y[i]:
            cost += b[i]*(pred[i]-y[i])
        else:
            cost += a[i]*(y[i]-pred[i])
    return nb_sample,cost

#验证集的真实值
val_y = pd.read_csv('../data/val_y.csv')
val_y.rename(columns={'qty_alipay_njhs':'y'},inplace=True)

#单模型
xgb = pd.read_csv('../data/xgb.csv',header=None)
xgb.columns = ['item_id','store_code','xgb']

rule = pd.read_csv('../data/rule_1236530.12809.csv',header=None)
rule.columns = ['item_id','store_code','rule']

rf = pd.read_csv('../data/rf.csv',header=None)
rf.columns = ['item_id','store_code','rf']

gbrt = pd.read_csv('../data/gbrt.csv',header=None)
gbrt.columns = ['item_id','store_code','gbrt']


# In[529]:



hashmap = {1:'1',2:'2',3:'3',4:'4',5:'5',6:'all'}
#xgb.store_code = xgb.store_code.apply(lambda x:hashmap[x])
rf.store_code = rf.store_code.apply(lambda x:hashmap[x])
gbrt.store_code = gbrt.store_code.apply(lambda x:hashmap[x])

#合并单模型
t = pd.merge(xgb,rf,on=['item_id','store_code'],how='outer')
t = pd.merge(t,rule,on=['item_id','store_code'],how='outer')
t = pd.merge(t,gbrt,on=['item_id','store_code'],how='outer')

t.fillna(0.0,inplace=True)


#如果预测值为负的，置为0
t.xgb = t.xgb.apply(lambda x:max(x,0.0))
t.rf = t.rf.apply(lambda x:max(x,0.0))
t.rule = t.rule.apply(lambda x:max(x,0.0))
t.gbrt = t.gbrt.apply(lambda x:max(x,0.0))


a_b = pd.read_csv('../data/config2.csv')
t = pd.merge(t,a_b,on=['item_id','store_code'])

col_num = t.shape[1]



#算出单模型预测值的最大、次大、最小、次小值
t['min_pred'] = 0
t['min2_pred'] = 0
t['max_pred'] = 0
t['max2_pred'] = 0
t['min_pred_norule'] = 0
t['min2_pred_norule'] = 0
t['max_pred_norule'] = 0
t['max2_pred_norule'] = 0


for row in range(t.shape[0]):
    preds = [t.iloc[row,2],t.iloc[row,4],t.iloc[row,5],t.iloc[row,7]]
    preds_norule = [t.iloc[row,3],t.iloc[row,5],t.iloc[row,7]]
    preds.sort()
    preds_norule.sort()
    t.iloc[row,col_num] = preds[0]#min_pred
    t.iloc[row,col_num+1] = preds[1]#min2_pred
    t.iloc[row,col_num+2] = preds[-1]#max_pred
    t.iloc[row,col_num+3] = preds[-2]#max2_pred
    t.iloc[row,col_num+4] = preds_norule[0]#min_pred_norule
    t.iloc[row,col_num+5] = preds_norule[1]#min2_pred_norule
    t.iloc[row,col_num+6] = preds_norule[-1]#max_pred_norule
    t.iloc[row,col_num+7] = preds_norule[-2]#max2_pred_norule


# In[530]:

t.to_csv('../result/all_pred.csv',index=None)


# In[536]:

val_y_pred.head(5)


# In[532]:

#根据补多、补少成本进行模型融合
t = pd.read_csv('../data/all_pred.csv')
col_num = t.shape[1]

t['pred'] = 0
for row in range(t.shape[0]):
    a = t.iloc[row,12]
    b = t.iloc[row,13]
    
    min_pred_norule = t.iloc[row,14]
    min2_pred_norule = t.iloc[row,15]
    max_pred_norule = t.iloc[row,16]
    max2_pred_norule = t.iloc[row,17]
    this_store_code = t.iloc[row,1]

    if a>b:
        t.iloc[row,col_num] = max_pred_norule*1.6
    else:
        t.iloc[row,col_num] = min_pred_norule*0.9


t.pred = t.pred.apply(lambda x:max(x,0.0))


# In[533]:

t.pred = 0.85*t.pred + 0.15*t.rule


val_y_pred = pd.merge(val_y,t,on=['item_id','store_code'],how='left')
val_y_pred.fillna(0.0,inplace=True)


# In[537]:

t.to_csv('../data/all_fencang_pred.csv',index=None)


# In[534]:

val_y_pred_all = val_y_pred[val_y_pred.store_code=='all']
val_y_pred_1 = val_y_pred[val_y_pred.store_code=='1']
val_y_pred_2 = val_y_pred[val_y_pred.store_code=='2']
val_y_pred_3 = val_y_pred[val_y_pred.store_code=='3']
val_y_pred_4 = val_y_pred[val_y_pred.store_code=='4']
val_y_pred_5 = val_y_pred[val_y_pred.store_code=='5']
print "all cost:",cal_cost(val_y_pred_all.y,val_y_pred_all.pred,val_y_pred_all.a,val_y_pred_all.b)
print "1 cost:",cal_cost(val_y_pred_1.y,val_y_pred_1.pred,val_y_pred_1.a,val_y_pred_1.b)
print "2 cost:",cal_cost(val_y_pred_2.y,val_y_pred_2.pred,val_y_pred_2.a,val_y_pred_2.b)
print "3 cost:",cal_cost(val_y_pred_3.y,val_y_pred_3.pred,val_y_pred_3.a,val_y_pred_3.b)
print "4 cost:",cal_cost(val_y_pred_4.y,val_y_pred_4.pred,val_y_pred_4.a,val_y_pred_4.b)
print "5 cost:",cal_cost(val_y_pred_5.y,val_y_pred_5.pred,val_y_pred_5.a,val_y_pred_5.b)
print "total cost:",cal_cost(val_y_pred.y,val_y_pred.pred,val_y_pred.a,val_y_pred.b)


# In[538]:

#验证集上的每个item产生的代价，从高到低排序，保存为csv文件，test set在预测时会将这部分item的预测值替换为前两周销量
val_y_pred['error'] = (val_y_pred.pred - val_y_pred.y)/val_y_pred.y
val_y_pred['cost'] = 0


# In[540]:

val_y_pred.shape


# In[541]:

val_y_pred.head(5)


# In[543]:

for row in range(val_y_pred.shape[0]):
    a = val_y_pred.iloc[row,13]
    b = val_y_pred.iloc[row,14]
    y = val_y_pred.iloc[row,2] 
    pred = val_y_pred.iloc[row,18]
    if pred>=y:
        val_y_pred.iloc[row,20] = b*(pred-y)
    else:
        val_y_pred.iloc[row,20] = a*(y-pred)



# In[544]:

val_y_pred = val_y_pred[['item_id','store_code','a','b','y','pred','error','cost']]
for row in range(val_y_pred.shape[0]):
    if val_y_pred.iloc[row,4] == 0.0:
        val_y_pred.iloc[row,6] = -99999

val_y_pred.sort_values(by='cost',axis=0,ascending=False,inplace=True)
val_y_pred.to_csv('../data/04_09.csv',index=None)


# In[ ]:



