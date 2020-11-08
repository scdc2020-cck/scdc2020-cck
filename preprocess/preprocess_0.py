#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.feature_selection
import sys
np.set_printoptions(threshold=sys.maxsize)


# In[4]:


x_name = os.path.join(os.path.dirname(os.getcwd()), 'raw', f"{'cst_feat_jan'}.csv")
df_x = pd.read_csv(x_name, index_col=0)

y_name = os.path.join(os.path.dirname(os.getcwd()), 'raw', f"{'train'}.csv")
df_y = pd.read_csv(y_name, index_col=0)

v_name = os.path.join(os.path.dirname(os.getcwd()), 'raw', f"{'val'}.csv")
df_v = pd.read_csv(v_name, index_col=0)

df_yv = pd.concat([df_y, df_v])
df = pd.merge(df_x, df_yv, on='cst_id_di')


# In[5]:


X = df.drop(columns = ['MRC_ID_DI'], axis=1)
y = df['MRC_ID_DI']


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=1, stratify = y)


# In[7]:


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


# In[8]:


select = sklearn.feature_selection.SelectKBest(k=113)
selected_features = select.fit(X_train, y_train)
indices_selected = selected_features.get_support(indices=True)
colnames_selected = [X_train.columns[i] for i in indices_selected]

X_train = X_train[colnames_selected]


# In[9]:


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


# In[10]:


train = X_train
train['cst_id_di'] = y_train.index
train = train.set_index('cst_id_di')
train['MRC_ID_DI'] = y_train

X_train = train.drop(columns = ['MRC_ID_DI'], axis=1)
y_train = train['MRC_ID_DI']


# In[11]:


select = sklearn.feature_selection.SelectKBest(k=380)
selected_features = select.fit(X_train, y_train)
indices_selected = selected_features.get_support(indices=True)
colnames_selected = [X_train.columns[i] for i in indices_selected]

X_train = X_train[colnames_selected]


# In[12]:


X_test = X_test.astype(np.float16)
combos = list(combinations(list(X_test.columns), 2))
colnames = list(X_test.columns) + ['_'.join(x) for x in combos]

poly = PolynomialFeatures(interaction_only=True, include_bias=False)
X_test = poly.fit_transform(X_test)
X_test = pd.DataFrame(X_test)
X_test.columns = colnames


noint_indicies = [i for i, x in enumerate(list((X_test == 0).all())) if x]
X_test = X_test.drop(X_test.columns[noint_indicies], axis = 1)


# In[13]:


test = X_test
test['cst_id_di'] = y_test.index
test = test.set_index('cst_id_di')
test['MRC_ID_DI'] = y_test

X_test = test.drop(columns = ['MRC_ID_DI'], axis=1)
y_test = test['MRC_ID_DI']

X_test = X_test[colnames_selected]


# In[14]:


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1, stratify = y_train)


# In[15]:


#X_train[X_train.columns] = (X_train[X_train.columns]+1)/2
#X_val[X_val.columns] = (X_val[X_val.columns]+1)/2


# In[16]:


df_train = pd.DataFrame(columns = ['cst_id_di', 'MRC_ID_DI', 'image'])
df_val = pd.DataFrame(columns = ['cst_id_di', 'MRC_ID_DI', 'image'])
df_test = pd.DataFrame(columns = ['cst_id_di', 'MRC_ID_DI', 'image'])


# In[17]:


for ind, k in enumerate(y_train.index):
    print(ind)
    an_array = X_train.iloc[ind].values
    new_row = {'cst_id_di':k, 'MRC_ID_DI':y_train.iloc[ind], 'image':an_array}
    df_train = df_train.append(new_row, ignore_index=True)


# In[18]:


df_train.to_csv('train_preprocess.csv')

del df_train


# In[19]:


for ind, k in enumerate(y_val.index):
    print(ind)    
    an_array = X_val.iloc[ind].values
    new_row = {'cst_id_di':k, 'MRC_ID_DI':y_val.iloc[ind], 'image':an_array}
    df_val = df_val.append(new_row, ignore_index=True)


# In[20]:


df_val.to_csv('val_preprocess.csv')

del df_val


# In[21]:


for ind, k in enumerate(y_test.index):
    print(ind)    
    an_array = X_test.iloc[ind].values
    new_row = {'cst_id_di':k, 'MRC_ID_DI':y_test.iloc[ind], 'image':an_array}
    df_test = df_test.append(new_row, ignore_index=True)


# In[22]:


df_test.to_csv('test_preprocess.csv')

del df_test


# In[ ]:


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
    
    for ind, k in enumerate(ty.index):
        print(ind)
        an_array = tx.iloc[ind].values
        new_row = {'cst_id_di':k, 'MRC_ID_DI':ty.iloc[ind], 'image':an_array}
        df_all = df_all.append(new_row, ignore_index=True)
    
    df_all.to_csv('all_preprocess_' + str(i) + '.csv')
    del df_all

