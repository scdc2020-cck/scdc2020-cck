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


df_img_name = os.path.join(os.getcwd(), 'preprocess', f"{'quiz_preprocess'}.csv")
df_img = pd.read_csv(df_img_name)


# In[ ]:


df_quiz_name = os.path.join(os.getcwd(), 'raw', f"{'quiz'}.csv")
df_quiz = pd.read_csv(df_quiz_name)


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


PRED = list()

for i in range(len(df_quiz)):
    X = returnX(df_imagedf.iloc[df_quiz.index[i]]["image"])
    y = df_quiz.loc[i, 'MRC_ID_DI']
    pred = model.predict(X)
    PRED.append(pred[y])
    
df_quiz['pred'] = PRED 


# In[ ]:


df_quiz.to_csv('quiz_s.csv')


# In[ ]:




