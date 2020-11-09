#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import os


# In[ ]:


#all_preprocess_0 ~ all_preprocess_9 를 합칠 때 실행

df = pd.DataFrame()
df_list = list()

for i in range(10):
    fname = os.path.join(os.getcwd(), f"{'all_preprocess_'+str(i)}.csv")
    dff = pd.read_csv(fname, index_col=0)
    df_list.append(dff)
    
df = pd.concat(df_list, ignore_index=True)
df.to_csv('all_preprocess.csv')


# In[ ]:


#quiz_preprocess_0 ~ quiz_preprocess_9 를 합칠 때 실행

df = pd.DataFrame()
df_list = list()

for i in range(10):
    fname = os.path.join(os.getcwd(), f"{'quiz_preprocess_'+str(i)}.csv")
    dff = pd.read_csv(fname, index_col=0)
    df_list.append(dff)
    
df = pd.concat(df_list, ignore_index=True)
df.to_csv('quiz_preprocess.csv')

