#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


import sys
ipython = get_ipython()

def hide_traceback(exc_tuple=None, filename=None, tb_offset=None,
                      exception_only=False, running_compiled_code=False):
       etype, value, tb = sys.exc_info()
       return ipython._showtraceback(etype, value, ipython.InteractiveTB.get_exception_only(etype, value))

ipython.showtraceback = hide_traceback


# In[3]:


p = 380

img_rows, img_cols, img_channel = p, p, 3

base_model = tf.keras.applications.Xception(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, img_channel))


# In[4]:


"""
def returnXandY(dff):
    df = dff.copy()
    df["image"] = df["image"].apply(lambda x: x.replace("[", "")).apply(lambda x: x.replace("]", ""))
    df['image'] = df['image'].apply(lambda x: np.fromstring(x, sep = " "))
    df['row_array'] = df['image'].apply(lambda x: np.resize(x, (p, 1))) + 1
    df['col_array'] = df['image'].apply(lambda x: np.resize(x,(1,p))) + 1
    df['final'] = df.apply(lambda x: np.dot(x.row_array, x.col_array), axis=1) / 4
    df = df.drop(columns=['Unnamed: 0', 'image', 'row_array', 'col_array'])
    df['final'] = df['final'].apply(lambda x: x.astype(np.float16))
    df['final'] = df['final'].apply(lambda x: np.repeat(x.flatten(), 3))
    df['final'] = df['final'].apply(lambda x: x.reshape((p, p, 3)))
    df['final'] = df['final'].apply(lambda x: np.expand_dims(x, axis=0))
    X = np.vstack(df['final'])
    y = df['MRC_ID_DI'].to_numpy()
    return X, y
"""


# In[5]:


def returnXandY(dff):
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
    y = df['MRC_ID_DI'].to_numpy()
    return X, y


# In[6]:


"""
def returnXandY(dff):
    df = dff.copy()
    df["image"] = df["image"].apply(lambda x: x.replace("[", "")).apply(lambda x: x.replace("]", ""))
    df['image'] = df['image'].apply(lambda x: np.fromstring(x, sep = " "))
    df['row_array'] = df['image'].apply(lambda x: np.tile(x, (p, 1)))
    df['col_array'] = df['image'].apply(lambda x: np.transpose([x] * p))
    df['final'] = ((df['row_array'] + df['col_array']) + 2) / 4
    df = df.drop(columns=['Unnamed: 0', 'image', 'row_array', 'col_array'])
    df['final'] = df['final'].apply(lambda x: x.astype(np.float16))
    df['final'] = df['final'].apply(lambda x: np.repeat(x.flatten(), 3))
    df['final'] = df['final'].apply(lambda x: x.reshape((p, p, 3)))
    df['final'] = df['final'].apply(lambda x: np.expand_dims(x, axis=0))
    X = np.vstack(df['final'])
    y = df['MRC_ID_DI'].to_numpy()
    return X, y
"""


# In[7]:


train_name = os.path.join(os.path.dirname(os.getcwd()), 'preprocess', f"{'train_preprocess'}.csv")
val_name = os.path.join(os.path.dirname(os.getcwd()), 'preprocess', f"{'val_preprocess'}.csv")
test_name = os.path.join(os.path.dirname(os.getcwd()), 'preprocess', f"{'test_preprocess'}.csv")


df_train = pd.read_csv(train_name)
df_val = pd.read_csv(val_name)
df_test = pd.read_csv(test_name)


# In[8]:


X_train, y_train = returnXandY(df_train)


# In[9]:


X_val, y_val = returnXandY(df_val)


# In[10]:


X_test, y_test = returnXandY(df_test)


# In[11]:


y_train = tf.keras.utils.to_categorical(y_train)
y_val = tf.keras.utils.to_categorical(y_val)
y_test = tf.keras.utils.to_categorical(y_test)


# In[12]:


@tf.function(experimental_relax_shapes=True)
def gelu(x):
    return 0.5 * x * (1 + tf.keras.backend.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))


add_model = tf.keras.Sequential()
#add_model.add(tf.keras.layers.GlobalAveragePooling2D(input_shape=base_model.output_shape[1:]))
add_model.add(tf.keras.layers.Flatten())
add_model.add(tf.keras.layers.Dropout(rate = 0.8))
#add_model.add(tf.keras.layers.Dense(units=512, activation=gelu))
#add_model.add(tf.keras.layers.Dropout(rate = 0.2))
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

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

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

plt.title('model auc')
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


final_score = (avg_lift)*0.7 + (avg_auroc)*0.3
print(avg_lift, avg_auroc, final_score)


# In[ ]:


#model.save('model.h5')


# In[ ]:


#df_all = pd.read_csv('0_all')

