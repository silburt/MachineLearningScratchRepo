import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.models import load_model
import tensorflow as tf

if __name__ == '__main__':
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True)) #check gpu is being used

    X, y = np.load('X.npy'), np.load('y.npy')

    model = Sequential()
    model.add(LSTM(512, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    checkpoint = ModelCheckpoint(dir_model, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    model.fit(X, y, epochs=epochs, batch_size=128, validation_split=0.2, callbacks=callbacks_list)
    model.save('model.h5')
