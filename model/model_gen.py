#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

seed_value= 0
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


# In[ ]:


#Transfer Learning

p = 380

img_rows, img_cols, img_channel = p, p, 3

base_model = tf.keras.applications.Xception(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, img_channel))


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
    df = df.drop(columns=['Unnamed: 0', 'image', 'row_array', 'col_array'])
    df['final'] = df['final'].apply(lambda x: x.astype(np.float16))
    df['final'] = df['final'].apply(lambda x: np.repeat(x.flatten(), 3))
    df['final'] = df['final'].apply(lambda x: x.reshape((p, p, 3)))
    df['final'] = df['final'].apply(lambda x: np.expand_dims(x, axis=0))
    X = np.vstack(df['final'])
    y = tf.keras.utils.to_categorical(df['MRC_ID_DI'].to_numpy())
    return X, y


# In[ ]:


train_name = os.path.join(os.path.dirname(os.getcwd()), 'preprocess', f"{'train_preprocess'}.csv")
val_name = os.path.join(os.path.dirname(os.getcwd()), 'preprocess', f"{'val_preprocess'}.csv")
test_name = os.path.join(os.path.dirname(os.getcwd()), 'preprocess', f"{'test_preprocess'}.csv")


df_train = pd.read_csv(train_name)
df_val = pd.read_csv(val_name)
df_test = pd.read_csv(test_name)


# In[ ]:


X_train, y_train = returnXandY(df_train)


# In[ ]:


X_val, y_val = returnXandY(df_val)


# In[ ]:


X_test, y_test = returnXandY(df_test)


# In[ ]:


add_model = tf.keras.Sequential()
add_model.add(tf.keras.layers.Flatten())
add_model.add(tf.keras.layers.Dropout(rate = 0.8)) # 오버피팅 방지
add_model.add(tf.keras.layers.Dense(units=512, activation=tf.nn.relu))
add_model.add(tf.keras.layers.Dropout(rate = 0.2)) # 오버피팅 방지
add_model.add(tf.keras.layers.Dense(units=11, activation=tf.nn.softmax))

model = tf.keras.Model(inputs=base_model.input, outputs=add_model(base_model.output))
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=1e-6),
              metrics=['accuracy'])


# In[ ]:


batch_size = 32
epochs = 40

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(dtype=np.float16)
train_datagen.fit(X_train)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(dtype=np.float16)
val_datagen.fit(X_val)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(dtype=np.float16)
test_datagen.fit(X_test)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3) #오버피팅 방지

history = model.fit_generator(
    train_datagen.flow(X_train, y_train, batch_size=batch_size),
    steps_per_epoch=X_train.shape[0] // batch_size,
    epochs=epochs,
    validation_data=val_datagen.flow(X_val, y_val, batch_size=batch_size),
    callbacks=[callback]
)


# In[ ]:


import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('model acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()


# In[ ]:


score = model.evaluate(test_datagen.flow(X_test, y_test, batch_size=batch_size))


# In[ ]:


preds = model.predict(test_datagen.flow(X_test, batch_size=batch_size, shuffle=False))


# In[ ]:


def LIFT(preds, y_test, cls): # >=2.5

    n_score = preds[:, cls]
    ind = np.argsort(-n_score)[:len(n_score)//5] #예측 score 상위 20%
    
    y_20 = y_test[ind]
    y_20_flat = y_20[:, cls]
    y_20_final = np.count_nonzero(y_20_flat == 1)
    
    y_test_flat = np.argmax(y_test, axis=1)
    y_test_final = y_test_flat[y_test_flat == cls] 
    lift = (y_20_final/len(y_20))/ (len(y_test_final)/len(y_test))
    print('LIFT Accuracy: ',  lift)
    return lift


# In[ ]:


lift_score = [0, 0, 0]
lift_score[0] = LIFT(preds, y_test, 3)
lift_score[1] = LIFT(preds, y_test, 4)
lift_score[2] = LIFT(preds, y_test, 7)
avg_lift = sum(lift_score) / 3


# In[ ]:


from sklearn.metrics import roc_curve
from sklearn.metrics import auc

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in [3, 4, 7]:
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], preds[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
avg_auroc = sum([roc_auc[3], roc_auc[4], roc_auc[7]]) / 3


# In[ ]:


final_score = (avg_lift/5)*0.7 + (avg_auroc)*0.3
print(avg_lift, avg_auroc, final_score)


# In[ ]:


model.save('final_model.h5')

