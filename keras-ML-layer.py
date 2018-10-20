import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Activation,Dropout
from keras import optimizers
import numpy as np
x_train = np.random.random((1000, 20))
y_train = to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
#to_categorical(y,num_class,dtype)
x_test = np.random.random((100, 20))
y_test = to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
model = Sequential()
model.add(Dense(64, activation='relu',input_dim = 20))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu',input_dim = 20))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))
model.compile(loss = 'categorical_crossentropy',
             optimizer = 'adam',
             metrics=['accuracy'])
model.fit(x_train,y_train,epochs = 20, batch_size = 1000)
score = model.evaluate(x_test,y_test,batch_size = 1000)
