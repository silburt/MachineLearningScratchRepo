#https://danijar.com/tips-for-training-recurrent-neural-networks/
#https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py - apparently this works...
#https://github.com/0xnurl/keras_character_based_ner/tree/master/src
#https://github.com/keras-team/keras/issues/197 - a lot of good stuff in here.
#https://github.com/mineshmathew/char_rnn_karpathy_keras/blob/master/char_rnn_of_karpathy_keras.py - suggested replacement network
#https://github.com/yxtay/char-rnn-text-generation/blob/master/keras_model.py - suggested replacement network

#https://groups.google.com/d/msg/keras-users/Y_FG_YEkjXs/QGC58mGHiU8J
# Reason not to use embedding layer for char-RNN - Yes: we want the output to be a probability distribution over characters. If each character was encoded by a dense vector learned with an Embedding layer, then output sampling would become a K-nearest neighbors problem over the embedding space, which would be much more complex to deal with than a dictionary lookup. But, if you're only using the embedding for the input but you're mapping the output with a simple dictionary, then you'll be fine (i.e. your output is still a probability distribution over characters).

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Embedding, TimeDistributed
#from keras.layers import Lambda, Input
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
from sklearn.model_selection import train_test_split
import tensorflow as tf
import sys

def train_model(genre, dir_model, seq_length, batch_size, epochs=100):
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True)) #check gpu is being used
    
    text_to_int, int_to_text, n_chars = np.load('playlists/%s/ancillary_char.npy'%genre)
    X = np.load('playlists/%s/X_sl%d_char.npy'%(genre,seq_length))
    y = np.load('playlists/%s/y_sl%d_char.npy'%(genre,seq_length))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    try:
        model = load_model(dir_model)
        print("successfully loaded previous model, continuing to train")
    except:
        print("generating new model")
        model = Sequential()
        
        model.add(GRU(256, dropout=0.2, recurrent_dropout=0.2, return_sequences=True,
                      stateful=True, input_shape=(X.shape[1], X.shape[2])))
        # output shape: (batch_size, seq_len, vocab_size)
        model.add(TimeDistributed(Dense(y.shape[2], activation='softmax')))
        loss = 'categorical_crossentropy'

        lr = 1e-3
        #decay = lr/epochs
        #optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay, clipvalue=1)
        optimizer = RMSprop(lr=lr)
        model.compile(loss=loss, optimizer=optimizer)

    print(model.summary())
    checkpoint = ModelCheckpoint(dir_model, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
              callbacks=callbacks_list, validation_data=(X_test, y_test), verbose=1)
    model.save(dir_model)

if __name__ == '__main__':
    genre = 'pop-rock-edm'
    seq_length = int(sys.argv[1])
    batch_size = 256
    
    dir_model = 'models/%s_sl%d_char.h5'%(genre,seq_length)
    
    train_model(genre,dir_model,seq_length,batch_size)



