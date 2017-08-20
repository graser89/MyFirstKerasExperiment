# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
from keras.models import Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.models import model_from_yaml


data_X1= pd.read_csv('data/predict_V/data1_51.csv', sep=';',decimal=',')
data_Y1= pd.read_csv('data/predict_V/data1_52.csv', sep=';',decimal=',')
data_X2=pd.read_csv('data/predict_V/data1_61.csv', sep=';',decimal=',')
data_Y2=pd.read_csv('data/predict_V/data1_62.csv', sep=';',decimal=',')

print(len(data_X1))

X_train=data_X1.values[:2190,:]
Y_train=data_Y1.values[:2190,:]

X_test=data_X2.values[:2190,:]
Y_test=data_Y2.values[:2190,:]



model = Sequential()
model.add(Dense(24, input_dim=24 ))
model.add(Dense(28,activation= 'relu' ))# activation= 'sigmoid'
model.add(Dense(4, activation= 'sigmoid'))

#хорошо бы выделить формированеие и обучени модели в отдельный метод на вход список оптимайзеров и функций потерь 
# и надо ли сохранять результаты по файлам  и сводные результаты
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])

#yaml_string = model.to_yaml()
#print(yaml_string)

history = model.fit(X_train, Y_train, batch_size=30, epochs=200,validation_split=0.8, validation_data=(X_test, Y_test), shuffle=True,initial_epoch=0)

result=model.predict(X_test, batch_size=32, verbose=0)
#result.tofile('data/predict_V/result1.csv',sep=';',format='%10.6f')
#Y_test.tofile('data/predict_V/result2.csv',sep=';',format='%10.6f')
df1 = pd.DataFrame(result)
df1.to_csv('data/predict_V/result1.csv',sep=';')

df2 = pd.DataFrame(Y_test)
df2.to_csv('data/predict_V/result2.csv',sep=';')

print(result)
print(Y_test)

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()

plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()

