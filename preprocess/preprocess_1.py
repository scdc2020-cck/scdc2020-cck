#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.feature_selection
import sys
np.set_printoptions(threshold=sys.maxsize)


# In[ ]:


df_name = os.path.join(os.path.dirname(os.getcwd()), 'raw', f"{'cst_feat_feb_quiz'}.csv")
df = pd.read_csv(df_name, index_col=0)


# In[ ]:


nc_name = os.path.join(os.path.dirname(os.getcwd()), 'raw', f"{'[Track1_데이터4] variable_dtype'}.xlsx")
nc = pd.read_excel(nc_name, index_col=0)


# In[ ]:


#Outlier처리: numerical데이터 중 전체 분포의 99%보다 크거나 1%보다 작은 값을 가질 경우 50% 값으로 변경

for i in df.columns[0:-1]:
    if nc.loc[i, 'dType'] == 'numerical':
        d_90 = df[i].quantile(0.99)
        d_10 = df[i].quantile(0.01)
        d_50 = df[i].quantile(0.50)
        df[i] = np.where(df[i] > d_90, d_90, df[i])
        df[i] = np.where(df[i] < d_10, d_10, df[i])


# In[ ]:


#colnames_selected.csv에 저장된 칼럼명을 활용하여 모델을 train_preprocess와 동일하게 quiz_preprocess생성.

df_col = pd.read_csv('colnames_selected.csv', delimiter=',', header=None)
colnames_selected = np.array(df_col[0].tolist())


# In[ ]:


#OOM Error방지를 위해 10개의 파일로 분할하여 생성.

from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures

td = [pd.DataFrame() for i in range(10)]
td[0], td[1], td[2], td[3], td[4], td[5], td[6], td[7], td[8], td[9] = np.array_split(df, 10)

for i, t in enumerate(td):
    
    df_all = pd.DataFrame(columns = ['cst_id_di', 'MRC_ID_DI', 'image'])
    
    t['MRC_ID_DI'] = 0
    tx = t.drop(columns = ['MRC_ID_DI'], axis=1)
    ty = t['MRC_ID_DI']
    
    tx = tx.astype(np.float16)
    combos = list(combinations(list(tx.columns), 2))
    colnames = list(tx.columns) + ['_'.join(x) for x in combos]

    poly = PolynomialFeatures(interaction_only=True, include_bias=False)
    tx = poly.fit_transform(tx)
    tx = pd.DataFrame(tx)
    tx.columns = colnames

    noint_indicies = [i for i, x in enumerate(list((tx == 0).all())) if x]
    tx = tx.drop(tx.columns[noint_indicies], axis = 1)
    
    tt = tx
    tt['cst_id_di'] = ty.index
    tt = tt.set_index('cst_id_di')
    tt['MRC_ID_DI'] = ty

    tx = tt.drop(columns = ['MRC_ID_DI'], axis=1)
    ty = tt['MRC_ID_DI']

    tx = tx[colnames_selected]
    
    for ind, k in enumerate(ty.index):
        
        an_array = tx.iloc[ind].values
        new_row = {'cst_id_di':k, 'image':an_array}
        df_all = df_all.append(new_row, ignore_index=True)
    
    df_all.to_csv('quiz_preprocess_' + str(i) + '.csv')
    del df_all

