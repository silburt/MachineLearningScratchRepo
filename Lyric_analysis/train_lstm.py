#https://github.com/vlraik/word-level-rnn-keras/blob/master/lstm_text_generation.py

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

from clean_lyrics import *

def process_song(song_dir, word_or_character):
    song = open(song_dir,'r',encoding='utf-8').read().lower()
    song = unidecode.unidecode(unicodetoascii(song, word_or_character))
    if word_or_character == 'word':
        return song.split()
    return song

def main(dir_lyrics,dir_model,n_songs,seq_length,epochs,word_or_character,train=0):
    files = glob.glob('%s*.txt'%dir_lyrics)[0:n_songs]
    songs, n_songs = [], len(files)
    for i,f in enumerate(files):
        songs.append(process_song(f, word_or_character))

    if word_or_character == 'character':
        data = sorted(list(set(' '.join(songs))))
    elif word_or_character == 'word':
        data = []
        for s in songs:
            data += s

    n_chars = len(data)
    char_to_int = dict((c, i) for i, c in enumerate(data))
    int_to_char = dict((i, c) for i, c in enumerate(data))
#    print(int_to_char)
#    print(char_to_int)

    dataX, dataY = [], []
    for i in range(n_songs):
        lyric = songs[i]
        for j in range(0,len(lyric)-seq_length, 1):
            seq_in = lyric[j:j + seq_length]
            seq_out = lyric[j + seq_length]
            print(seq_in, seq_out)
            dataX.append([char_to_int[char] for char in seq_in])
            dataY.append(char_to_int[seq_out])
    n_patterns = len(dataX)
    print("Total Patterns: ", n_patterns)

    # prepare
    X = np.reshape(dataX, (n_patterns,seq_length,1))    # reshape X:[samples,time steps,features]
    X = X / float(n_chars)                              # normalize
    y = np_utils.to_categorical(dataY)                  # 1-hot encode the output variable
    print(y.shape)

    save = 1
    if save == 1:
        np.save('X.npy',X)
        np.save('y.npy',y)

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

            model.fit(X, y, epochs=epochs, batch_size=128, validation_split=0.2, callbacks=callbacks_list)
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
        #index = np.argmax(pred)
        index = np.random.choice(len(pred[0]), p=pred[0])
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    print("\nDone.")

if __name__ == '__main__':
    n_songs = 200
    seq_length = 35
    epochs = 1
    word_or_character = 'character'
    
    #genre = sys.argv[1]
    genre = 'country'
    dir_lyrics = 'playlists/%s/'%genre
    dir_model = 'models/%s_sl%d_%s.h5'%(genre,seq_length,word_or_character)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True)) #check gpu is being used
    main(dir_lyrics,dir_model,n_songs,seq_length,epochs,word_or_character)
