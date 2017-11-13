import numpy as np
import glob
import sys

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.models import load_model

def unicodetoascii(text):
    TEXT = (text.
            replace('\xe2\x80\x99', "'").
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
            replace('\xc3\xa0','a').
            replace('\xc3\xa4','a').
            replace('\xc3\xb1','n').
            replace('\xc3\xb3','o').
            replace('_',' ').
            replace('*',' ').
            replace('+','and').
            replace('{','[').
            replace('}',']').
            replace('`',"'").
            replace('"',"'").
            replace('$','').
            replace('&','and').
            replace('/',' and ')
            )
    return TEXT

def process_song(song_dir):
    song = unicodetoascii(open(song_dir,'r').read().lower())
    return song

def main(dir_lyrics,dir_model,n_songs,seq_length,epochs,train=0):
    files = glob.glob('%s*.txt'%dir_lyrics)[0:n_songs]
    songs = []
    for i,f in enumerate(files):
        songs.append(process_song(f))

    chars = sorted(list(set(' '.join(songs))))
    n_chars = len(chars)
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))

    dataX = []
    dataY = []
    for i in range(n_songs):
        song_text = songs[i]
        for j in range(0,len(song_text)-seq_length, 1):
            seq_in = song_text[j:j + seq_length]
            seq_out = song_text[j + seq_length]
            #print repr(seq_in),"<<>>", repr(seq_out)
            dataX.append([char_to_int[char] for char in seq_in])
            dataY.append(char_to_int[seq_out])
    n_patterns = len(dataX)
    print "Total Patterns: ", n_patterns

    # reshape X to be [samples, time steps, features]
    X = np.reshape(dataX, (n_patterns, seq_length, 1))
    # normalize
    X = X / float(n_chars)
    # one hot encode the output variable
    y = np_utils.to_categorical(dataY)

    try:
        model = model = load_model(dir_model)
        print "successfully loaded model"
    except:
        if train == 1:
            print "couldnt find model. Training... (this will take a while)"
            model = Sequential()
            model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
            model.add(Dropout(0.2))
            model.add(Dense(y.shape[1], activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam')

            model.fit(X, y, epochs=epochs, batch_size=128)
            model.save('models/model.h5')
            print "successfully trained and saved model"
        else:
            print "couldnt find model. Train=0. Exiting."
            return None

    # text generation
    start = np.random.randint(0, len(dataX)-1)
    pattern = dataX[start]
    print "Seed:"
    print "\"", ''.join([int_to_char[value] for value in pattern]), "\""
    # generate characters
    for i in range(1000):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_chars)
        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    print "\nDone."


if __name__ == '__main__':
    n_songs=-1
    seq_length = 30
    epochs = 6
    
    dir_lyrics = 'playlists/country/'
    dir_model = 'models/country.h5'

    main(dir_lyrics,dir_model,n_songs,seq_length,epochs)
