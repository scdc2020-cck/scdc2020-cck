#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.feature_selection
import sys
np.set_printoptions(threshold=sys.maxsize) #array의 데이터(380, )를 truncation없이 저장하기 위해


# In[ ]:


x_name = os.path.join(os.path.dirname(os.getcwd()), 'raw', f"{'cst_feat_jan'}.csv")
df_x = pd.read_csv(x_name, index_col=0)

y_name = os.path.join(os.path.dirname(os.getcwd()), 'raw', f"{'train'}.csv")
df_y = pd.read_csv(y_name, index_col=0)

v_name = os.path.join(os.path.dirname(os.getcwd()), 'raw', f"{'val'}.csv")
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


X = df.drop(columns = ['MRC_ID_DI'], axis=1)
y = df['MRC_ID_DI']


# In[ ]:


#stratify를 이용해 test데이터 MRC_ID_DI분포를 전체 데이터 MRC_ID_DI분포와 일치하게 함.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=1, stratify = y)


# In[ ]:


#train데이터의 각 MRC_ID_DI의 수를 동일하게 함. 

train = X_train
train['cst_id_di'] = y_train.index
train = train.set_index('cst_id_di')
train['MRC_ID_DI'] = y_train

train_0 = train[train['MRC_ID_DI'] == 0].sample(frac=1)
train_1 = train[train['MRC_ID_DI'] == 1].sample(frac=1)
train_2 = train[train['MRC_ID_DI'] == 2].sample(frac=1)
train_3 = train[train['MRC_ID_DI'] == 3].sample(frac=1)
train_4 = train[train['MRC_ID_DI'] == 4].sample(frac=1)
train_5 = train[train['MRC_ID_DI'] == 5].sample(frac=1)
train_6 = train[train['MRC_ID_DI'] == 6].sample(frac=1)
train_7 = train[train['MRC_ID_DI'] == 7].sample(frac=1)
train_8 = train[train['MRC_ID_DI'] == 8].sample(frac=1)
train_9 = train[train['MRC_ID_DI'] == 9].sample(frac=1)
train_10 = train[train['MRC_ID_DI'] == 10].sample(frac=1)

sample_size = min(len(train_0),len(train_1), len(train_2), len(train_3), len(train_4), len(train_5), len(train_6), len(train_7),
                 len(train_8), len(train_9), len(train_10))

train_f = pd.concat([train_0.head(sample_size), train_1.head(sample_size), train_2.head(sample_size), train_3.head(sample_size),
                    train_4.head(sample_size), train_5.head(sample_size), train_6.head(sample_size), train_7.head(sample_size), 
                    train_8.head(sample_size), train_9.head(sample_size), train_10.head(sample_size)]).sample(frac=1)

X_train = train_f.drop(columns = ['MRC_ID_DI'], axis=1)
y_train = train_f['MRC_ID_DI']


# In[ ]:


#column 수 변화: 226 -> 90 -> 4005 -> 380

select = sklearn.feature_selection.SelectKBest(k=90)
selected_features = select.fit(X_train, y_train)
indices_selected = selected_features.get_support(indices=True)
colnames_selected = [X_train.columns[i] for i in indices_selected]

X_train = X_train[colnames_selected]


# In[ ]:


from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures

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


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1, stratify = y_train)


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


#quiz_preprocess데이터를 final_model.h5를 train한 데이터와 동일하게 생성하기 위해 train데이터의 칼럼을 저장

np.savetxt("colnames_selected.csv", colnames_selected, delimiter =", ", fmt ='% s') 

