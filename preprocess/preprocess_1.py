#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
import sklearn.feature_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from itertools import combinations
import sys
np.set_printoptions(threshold=sys.maxsize) #array의 데이터(380, )를 truncation없이 저장하기 위해


# In[ ]:


x_name = os.path.join(os.getcwd(), 'raw', f"{'cst_feat_jan'}.csv")
df_x = pd.read_csv(x_name, index_col=0)

y_name = os.path.join(os.getcwd(), 'raw', f"{'train'}.csv")
df_y = pd.read_csv(y_name, index_col=0)

v_name = os.path.join(os.getcwd(), 'raw', f"{'val'}.csv")
df_v = pd.read_csv(v_name, index_col=0)

df_yv = pd.concat([df_y, df_v])
df = pd.merge(df_x, df_yv, on='cst_id_di')


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


# 1월 고객 예측 결과를 이용해 각 MRC_ID_DI의 예측 probability 상위 데이터를 선별하여 train데이터로 사용.
#  각 MRC_ID_DI의 train, val데이터 수는 모두 동일하게, test데이터는 전체 데이터의 분포와 일치하게 설정.

pred_name = os.path.join(os.path.dirname(os.getcwd()), 'predict', f"{'jan_pred'}.csv")
df_pred = pd.read_csv(pred_name)


# In[ ]:


train_0 = df_pred[df_pred['MRC_ID_DI'] == 0].sort_values(by="pred", ascending=False)[:474]
train_1 = df_pred[df_pred['MRC_ID_DI'] == 1].sort_values(by="pred", ascending=False)[:474]
train_2 = df_pred[df_pred['MRC_ID_DI'] == 2].sort_values(by="pred", ascending=False)[:474]
train_3 = df_pred[df_pred['MRC_ID_DI'] == 3].sort_values(by="pred", ascending=False)[:474]
train_4 = df_pred[df_pred['MRC_ID_DI'] == 4].sort_values(by="pred", ascending=False)[:474]
train_5 = df_pred[df_pred['MRC_ID_DI'] == 5].sort_values(by="pred", ascending=False)[:474]
train_6 = df_pred[df_pred['MRC_ID_DI'] == 6].sort_values(by="pred", ascending=False)[:474]
train_7 = df_pred[df_pred['MRC_ID_DI'] == 7].sort_values(by="pred", ascending=False)[:474]
train_8 = df_pred[df_pred['MRC_ID_DI'] == 8].sort_values(by="pred", ascending=False)[:474]
train_9 = df_pred[df_pred['MRC_ID_DI'] == 9].sort_values(by="pred", ascending=False)[:474]
train_10 = df_pred[df_pred['MRC_ID_DI'] == 10].sort_values(by="pred", ascending=False)[:474]


# In[ ]:


train_f = pd.concat([train_0, train_1, train_2, train_3, train_4, train_5, train_6, train_7, 
                    train_8, train_9, train_10]).sample(frac=1)

X_train = train_f.drop(columns = ['MRC_ID_DI'], axis=1)
y_train = train_f['MRC_ID_DI']


# In[ ]:


val_0 = df_pred[df_pred['MRC_ID_DI'] == 0].sort_values(by="pred", ascending=False)[474:526]
val_1 = df_pred[df_pred['MRC_ID_DI'] == 1].sort_values(by="pred", ascending=False)[474:526]
val_2 = df_pred[df_pred['MRC_ID_DI'] == 2].sort_values(by="pred", ascending=False)[474:526]
val_3 = df_pred[df_pred['MRC_ID_DI'] == 3].sort_values(by="pred", ascending=False)[474:526]
val_4 = df_pred[df_pred['MRC_ID_DI'] == 4].sort_values(by="pred", ascending=False)[474:526]
val_5 = df_pred[df_pred['MRC_ID_DI'] == 5].sort_values(by="pred", ascending=False)[474:526]
val_6 = df_pred[df_pred['MRC_ID_DI'] == 6].sort_values(by="pred", ascending=False)[474:526]
val_7 = df_pred[df_pred['MRC_ID_DI'] == 7].sort_values(by="pred", ascending=False)[474:526]
val_8 = df_pred[df_pred['MRC_ID_DI'] == 8].sort_values(by="pred", ascending=False)[474:526]
val_9 = df_pred[df_pred['MRC_ID_DI'] == 9].sort_values(by="pred", ascending=False)[474:526]
val_10 = df_pred[df_pred['MRC_ID_DI'] == 10].sort_values(by="pred", ascending=False)[474:526]


# In[ ]:


val_f = pd.concat([val_0, val_1, val_2, val_3, val_4, val_5, val_6, val_7, 
                    val_8, val_9, val_10]).sample(frac=1)

X_val = val_f.drop(columns = ['MRC_ID_DI'], axis=1)
y_val = val_f['MRC_ID_DI']


# In[ ]:


test_0 = df_pred[df_pred['MRC_ID_DI'] == 0].sort_values(by="pred", ascending=False)[:-len(df_pred[df_pred['MRC_ID_DI'] == 0])*0.01]
test_1 = df_pred[df_pred['MRC_ID_DI'] == 1].sort_values(by="pred", ascending=False)[:-len(df_pred[df_pred['MRC_ID_DI'] == 1])*0.01]
test_2 = df_pred[df_pred['MRC_ID_DI'] == 2].sort_values(by="pred", ascending=False)[:-len(df_pred[df_pred['MRC_ID_DI'] == 2])*0.01]
test_3 = df_pred[df_pred['MRC_ID_DI'] == 3].sort_values(by="pred", ascending=False)[:-len(df_pred[df_pred['MRC_ID_DI'] == 3])*0.01]
test_4 = df_pred[df_pred['MRC_ID_DI'] == 4].sort_values(by="pred", ascending=False)[:-len(df_pred[df_pred['MRC_ID_DI'] == 4])*0.01]
test_5 = df_pred[df_pred['MRC_ID_DI'] == 5].sort_values(by="pred", ascending=False)[:-len(df_pred[df_pred['MRC_ID_DI'] == 5])*0.01]
test_6 = df_pred[df_pred['MRC_ID_DI'] == 6].sort_values(by="pred", ascending=False)[:-len(df_pred[df_pred['MRC_ID_DI'] == 6])*0.01]
test_7 = df_pred[df_pred['MRC_ID_DI'] == 7].sort_values(by="pred", ascending=False)[:-len(df_pred[df_pred['MRC_ID_DI'] == 7])*0.01]
test_8 = df_pred[df_pred['MRC_ID_DI'] == 8].sort_values(by="pred", ascending=False)[:-len(df_pred[df_pred['MRC_ID_DI'] == 8])*0.01]
test_9 = df_pred[df_pred['MRC_ID_DI'] == 9].sort_values(by="pred", ascending=False)[:-len(df_pred[df_pred['MRC_ID_DI'] == 9])*0.01]
test_10 = df_pred[df_pred['MRC_ID_DI'] == 10].sort_values(by="pred", ascending=False)[:-len(df_pred[df_pred['MRC_ID_DI'] == 10])*0.01]


# In[ ]:


test_f = pd.concat([test_0, test_1, test_2, test_3, test_4, test_5, test_6, test_7, 
                    test_8, test_9, test_10]).sample(frac=1)

X_test = test_f.drop(columns = ['MRC_ID_DI'], axis=1)
y_test = test_f['MRC_ID_DI']


# In[ ]:


#column 수 변화: 226 -> 90 -> 4005 -> 380

select = sklearn.feature_selection.SelectKBest(k=90)
selected_features = select.fit(X_train, y_train)
indices_selected = selected_features.get_support(indices=True)
colnames_selected = [X_train.columns[i] for i in indices_selected]

X_train = X_train[colnames_selected]


# In[ ]:


X_train = X_train.astype(np.float16)
combos = list(combinations(list(X_train.columns), 2))
colnames = list(X_train.columns) + ['_'.join(x) for x in combos]

poly = PolynomialFeatures(interaction_only=True, include_bias=False)
X_train = poly.fit_transform(X_train)
X_train = pd.DataFrame(X_train)
X_train.columns = colnames


noint_indicies = [i for i, x in enumerate(list((X_train == 0).all())) if x]
X_train = X_train.drop(X_train.columns[noint_indicies], axis = 1)


# In[ ]:


train = X_train
train['cst_id_di'] = y_train.index
train = train.set_index('cst_id_di')
train['MRC_ID_DI'] = y_train

X_train = train.drop(columns = ['MRC_ID_DI'], axis=1)
y_train = train['MRC_ID_DI']


# In[ ]:


select = sklearn.feature_selection.SelectKBest(k=380)
selected_features = select.fit(X_train, y_train)
indices_selected = selected_features.get_support(indices=True)
colnames_selected = [X_train.columns[i] for i in indices_selected]

X_train = X_train[colnames_selected] #colnames_selected를 이용해 val데이터와 test데이터를 train데이터와 동일하게 생성.


# In[ ]:


X_val = X_val.astype(np.float16)
combos = list(combinations(list(X_val.columns), 2))
colnames = list(X_val.columns) + ['_'.join(x) for x in combos]

poly = PolynomialFeatures(interaction_only=True, include_bias=False)
X_val = poly.fit_transform(X_val)
X_val = pd.DataFrame(X_val)
X_val.columns = colnames


noint_indicies = [i for i, x in enumerate(list((X_val == 0).all())) if x]
X_val = X_val.drop(X_val.columns[noint_indicies], axis = 1)


# In[ ]:


val = X_val
val['cst_id_di'] = y_val.index
val = val.set_index('cst_id_di')
val['MRC_ID_DI'] = y_val

X_val = val.drop(columns = ['MRC_ID_DI'], axis=1)
y_val = val['MRC_ID_DI']

X_val = X_val[colnames_selected]


# In[ ]:


X_test = X_test.astype(np.float16)
combos = list(combinations(list(X_test.columns), 2))
colnames = list(X_test.columns) + ['_'.join(x) for x in combos]

poly = PolynomialFeatures(interaction_only=True, include_bias=False)
X_test = poly.fit_transform(X_test)
X_test = pd.DataFrame(X_test)
X_test.columns = colnames


noint_indicies = [i for i, x in enumerate(list((X_test == 0).all())) if x]
X_test = X_test.drop(X_test.columns[noint_indicies], axis = 1)


# In[ ]:


test = X_test
test['cst_id_di'] = y_test.index
test = test.set_index('cst_id_di')
test['MRC_ID_DI'] = y_test

X_test = test.drop(columns = ['MRC_ID_DI'], axis=1)
y_test = test['MRC_ID_DI']

X_test = X_test[colnames_selected]


# In[ ]:


df_train = pd.DataFrame(columns = ['cst_id_di', 'MRC_ID_DI', 'image'])
df_val = pd.DataFrame(columns = ['cst_id_di', 'MRC_ID_DI', 'image'])
df_test = pd.DataFrame(columns = ['cst_id_di', 'MRC_ID_DI', 'image'])


# In[ ]:


for ind, k in enumerate(y_train.index):
    an_array = X_train.iloc[ind].values
    new_row = {'cst_id_di':k, 'MRC_ID_DI':y_train.iloc[ind], 'image':an_array}
    df_train = df_train.append(new_row, ignore_index=True)


# In[ ]:


df_train.to_csv('train_preprocess.csv')

del df_train


# In[ ]:


for ind, k in enumerate(y_val.index): 
    an_array = X_val.iloc[ind].values
    new_row = {'cst_id_di':k, 'MRC_ID_DI':y_val.iloc[ind], 'image':an_array}
    df_val = df_val.append(new_row, ignore_index=True)


# In[ ]:


df_val.to_csv('val_preprocess.csv')

del df_val


# In[ ]:


for ind, k in enumerate(y_test.index):  
    an_array = X_test.iloc[ind].values
    new_row = {'cst_id_di':k, 'MRC_ID_DI':y_test.iloc[ind], 'image':an_array}
    df_test = df_test.append(new_row, ignore_index=True)


# In[ ]:


df_test.to_csv('test_preprocess.csv')

del df_test


# In[ ]:


#OOM Error방지를 위해 10개의 파일로 분할하여 생성.

td = [pd.DataFrame() for i in range(10)]
td[0], td[1], td[2], td[3], td[4], td[5], td[6], td[7], td[8], td[9] = np.array_split(df, 10)

for i, t in enumerate(td):
    
    df_all = pd.DataFrame(columns = ['cst_id_di', 'MRC_ID_DI', 'image'])
    
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
    
    tx[tx.columns] = (tx[tx.columns]+1)/2

    for ind, k in enumerate(ty.index):
        print(ind)
        an_array = tx.iloc[ind].values
        new_row = {'cst_id_di':k, 'MRC_ID_DI':ty.iloc[ind], 'image':an_array}
        df_all = df_all.append(new_row, ignore_index=True)
    
    df_all.to_csv('all_preprocess_' + str(i) + '.csv')
    del df_all


# In[ ]:


#quiz_preprocess데이터를 final_model.h5를 train한 데이터와 동일하게 생성하기 위해 train데이터의 칼럼을 저장

np.savetxt("colnames_selected.csv", colnames_selected, delimiter =", ", fmt ='% s') 

