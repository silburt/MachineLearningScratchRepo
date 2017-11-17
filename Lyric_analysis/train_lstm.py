#https://github.com/vlraik/word-level-rnn-keras/blob/master/lstm_text_generation.py

#force GPU - https://github.com/fchollet/keras/issues/4613
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import glob
import sys
import unidecode

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.models import load_model
import tensorflow as tf

def unicodetoascii(text):
    TEXT = (text.
            replace('\xe2\x80\x99', "'").
            replace('\x92',"'").
            replace('\xe2\x80\x8be', 'e').
            replace('\xc3\xa9', 'e').
            replace('\xc2\x92',"'").
            replace('\xe2\x80\x90', '-').
            replace('\xe2\x80\x91', '-').
            replace('\xe2\x80\x92', '-').
            replace('\xe2\x80\x93', '-').
            replace('\xe2\x80\x94', '-').
            replace('\xe2\x80\x94', '-').
            replace('\xe2\x80\x98', "'").
            replace('\xe2\x80\x9b', "'").
            replace('\xe2\x80\x9c', '"').
            replace('\xe2\x80\x9c', '"').
            replace('\xe2\x80\x9d', '"').
            replace('\xe2\x80\x9e', '"').
            replace('\xe2\x80\x9f', '"').
            replace('\xe2\x80\xa6', '...').
            replace('\xe2\x80\xb2', "'").
            replace('\xe2\x80\xb3', "'").
            replace('\xe2\x80\xb4', "'").
            replace('\xe2\x80\xb5', "'").
            replace('\xe2\x80\xb6', "'").
            replace('\xe2\x80\xb7', "'").
            replace('\xe2\x81\xba', "+").
            replace('\xe2\x81\xbb', "-").
            replace('\xe2\x81\xbc', "=").
            replace('\xe2\x81\xbd', "(").
            replace('\xe2\x81\xbe', ")").
            replace('\xe2\x80\x8b', '').
            replace('\xc3\xa2\xe2\x82\xac\xcb\x9c',"'").
            replace('\xc3\xa4','a').
            replace('\xc3\xb1','n').
            replace('\xc3\xb3','o').
            replace('_',' ').
            replace('*',' ').
            replace('+','and').
            replace('{','(').
            replace('}',')').
            replace('[','(').
            replace(']',')').
            replace('`',"'").
            replace('"',"'").
            replace('$','').
            replace('&','and').
            replace('/',' and ')
            )
    return TEXT

def process_song(song_dir):
    song = open(song_dir,'r',encoding='utf-8').read().lower()
    return unidecode.unidecode(unicodetoascii(song))

def main(dir_lyrics,dir_model,n_songs,seq_length,epochs,train=1):
    files = glob.glob('%s*.txt'%dir_lyrics)[0:n_songs]
    songs, n_songs = [], len(files)
    for i,f in enumerate(files):
        songs.append(process_song(f))

    chars = sorted(list(set(' '.join(songs))))
    n_chars = len(chars)
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))

    dataX, dataY = [], []
    for i in range(n_songs):
        song_text = songs[i]
        for j in range(0,len(song_text)-seq_length, 1):
            seq_in = song_text[j:j + seq_length]
            seq_out = song_text[j + seq_length]
            dataX.append([char_to_int[char] for char in seq_in])
            dataY.append(char_to_int[seq_out])
    n_patterns = len(dataX)
    print("Total Patterns: ", n_patterns)

    # prepare
    X = np.reshape(dataX, (n_patterns,seq_length,1))    # reshape X:[samples,time steps,features]
    X = X / float(n_chars)                              # normalize
    y = np_utils.to_categorical(dataY)                  # 1-hot encode the output variable

    try:
        model = load_model(dir_model)
        print("successfully loaded model")
    except:
        if train == 1:
            print("couldnt find model. Training... (this will take a while)")
            model = Sequential()
            model.add(LSTM(512, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
            model.add(Dropout(0.2))
            model.add(LSTM(512, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(y.shape[1], activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam')
            checkpoint = ModelCheckpoint(dir_model, monitor='loss', verbose=1, save_best_only=True, mode='min')
            callbacks_list = [checkpoint]

            model.fit(X, y, epochs=epochs, batch_size=128, callbacks=callbacks_list)
            model.save(dir_model)
            print("successfully trained and saved model")
        else:
            print("couldnt find model and train=0. Exiting.")
            return None

    # text generation
    seed = np.random.randint(0, n_songs)
    ini = songs[seed][0:seq_length]     #set initial = start of a song
    pattern = [char_to_int[v] for v in list(ini)]
    print("Song:%s"%files[seed])
    print("Seed:%s"%ini)
    # generate characters
    for i in range(100):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_chars)
        pred = model.predict(x, verbose=0)
        #index = np.argmax(prediction)
        index = np.random.choice(len(pred[0]), p=pred[0])
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    print("\nDone.")

if __name__ == '__main__':
    n_songs= -1
    seq_length = 30
    epochs = 6
    
    #genre = sys.argv[1]
    genre = 'country'
    dir_lyrics = 'playlists/%s/'%genre
    dir_model = 'models/%s.h5'%genre

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True)) #check gpu is being used
    main(dir_lyrics,dir_model,n_songs,seq_length,epochs)
