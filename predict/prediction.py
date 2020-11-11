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
batch_size = 128
model_name = os.path.join(os.path.dirname(os.getcwd()), 'model', 'final_model.h5')
model = tf.keras.models.load_model(model_name)


# In[ ]:


df_img_name = os.path.join(os.path.dirname(os.getcwd()), 'preprocess', f"{'quiz_preprocess'}.csv")
df_img = pd.read_csv(df_img_name)


# In[ ]:


df_quiz_name = os.path.join(os.path.dirname(os.getcwd()), 'raw', f"{'quiz'}.csv")
df_quiz = pd.read_csv(df_quiz_name)


# In[ ]:


df_all = pd.merge(df_img.drop(columns=['Unnamed: 0', 'MRC_ID_DI']),df_quiz,how = 'left' ,on=['cst_id_di'])


# In[ ]:


def returnXandY(dff): 
    
    """
    [ 데이터의 형태(p, )를 이미지(1, p, p, 3)로 변환 ]
    Dataframe의 한 row를 row_array = (p, 1), col_array = (1, p) 로 resize한 뒤 
    np.dot(row_array, col_array)하여 (p, p)의 행렬 생성. 
    기존의 데이터 값이 -1 ~ 1 인 것을 고려하여, np.dot결과가 
    정규화된 이미지 값 0 ~ 1 이 되기위해 전체에 +1 을 한 뒤 /2를 함.
    np.repeat을 한 뒤 reshape, np.expand_dims을 통해 (p, p)=>(1, p, p, 3)
    
    *전처리 과정에서 각 row의 값을 하나씩 이미지로 계산하여 저장하는 것보다
    *train데이터를 로드하는 과정에서 DataFrame 전체를 한 번에 계산할 경우, 
    *계산 시간과 디스크 메모리(1월 데이터 기준 30GB -> 1GB)를 크게 절약할 수 있음.
    """ 
    
    df = dff.copy()
    df["image"] = df["image"].apply(lambda x: x.replace("[", "")).apply(lambda x: x.replace("]", ""))
    df['image'] = df['image'].apply(lambda x: np.fromstring(x, sep = " "))
    df['row_array'] = df['image'].apply(lambda x: np.resize(x, (p, 1)))
    df['col_array'] = df['image'].apply(lambda x: np.resize(x,(1,p)))
    df['final'] = (df.apply(lambda x: np.dot(x.row_array, x.col_array), axis=1) + 1) / 2
    df = df.drop(columns=['image', 'row_array', 'col_array'])
    df['final'] = df['final'].apply(lambda x: x.astype(np.float16))
    df['final'] = df['final'].apply(lambda x: np.repeat(x.flatten(), 3))
    df['final'] = df['final'].apply(lambda x: x.reshape((p, p, 3)))
    df['final'] = df['final'].apply(lambda x: np.expand_dims(x, axis=0))
    
    X = np.vstack(df['final'])
    y = df['MRC_ID_DI'].to_numpy()
    
    del df
    
    return X, y


# In[ ]:


import gc

td = [pd.DataFrame() for i in range(300)]
for i, sp in enumerate(np.array_split(df_all, 300)):
    td[i] = sp
    
Score = list()

for i, t in enumerate(td):
    
    print(i)
    
    X_quiz, y_quiz = returnXandY(t)
    
    quiz_datagen = tf.keras.preprocessing.image.ImageDataGenerator(dtype=np.float16)
    quiz_datagen.fit(X_quiz)
    
    quiz = model.predict(quiz_datagen.flow(X_quiz, batch_size=batch_size, shuffle=False))
    
    for i, y in enumerate(y_quiz):
        Score.append(quiz[i][y])
    
    tf.compat.v1.reset_default_graph()
    del X_quiz, y_quiz, quiz_datagen, quiz
    gc.collect()


# In[ ]:


df_quiz['Score'] = Score 


# In[ ]:


df_quiz.to_csv('quiz_s.csv', index=False)

