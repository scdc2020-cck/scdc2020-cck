#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import numpy as np
import tensorflow as tf
import random

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

seed_value= 0
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


# In[ ]:


p = 380
model = tf.keras.models.load_model('model/final_model.h5')


# In[ ]:


df_all_name = os.path.join(os.getcwd(), 'raw', 'preprocess', f"{'all_preprocess'}.csv")
df_all = pd.read_csv(df_all_name)


# In[ ]:


def returnX(X):
    
    """
    train데이터와 동일하게 DatraFrame의 row(p, )를 이미지(1, p, p, 3)로 변경
    """
    
    an_array = X.values
    t_array = np.transpose([an_array])
    final_array = (np.dot(np.resize(an_array, (p, 1)), np.resize(t_array,(1, p)), axis=1) +1) / 2
    final_array = final_array.astype(np.float16)
    final_array = np.repeat(final_array.flatten(), 3)
    data = final_array.reshape((p,p, 3))
    data = np.expand_dims(data, axis=0)
    return data


# In[ ]:


#1차 preprocess모델로 생성한 모델을 이용해, 1월 고객을 predict하여 정답 레이블의 probability를 저장.

PRED = list()

for i in range(len(df_all)):
    X = returnX(df_all.loc[i, 'image'])
    y = df_all.loc[i, 'MRC_ID_DI']
    pred = model.predict(X)
    PRED.append(pred[y])
    
df_all['pred'] = PRED 


# In[ ]:


df_all = df_all.drop(columns=['image'])


# In[ ]:


df_all.to_csv('jan_pred.csv')

