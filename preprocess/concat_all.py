#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import os


# In[ ]:


d = 'image31'
n = '0'

df = pd.DataFrame()
df_list = list()

for i in range(10):
    fname = os.path.join(os.getcwd(), f"{d +'_'+n+'_'+'all_'+str(i)}.csv")
    dff = pd.read_csv(fname, index_col=0)
    df_list.append(dff)
    
df = pd.concat(df_list, ignore_index=True)
df.to_csv(d +'_'+n+'_'+'all.csv')


# In[ ]:




