#https://github.com/vlraik/word-level-rnn-keras/blob/master/lstm_text_generation.py

import numpy as np
import glob
import sys
from keras.utils import np_utils
from process_lyrics import *

def main(genre,n_songs,seq_length,word_or_character):
    dir_lyrics = 'playlists/%s/'%genre
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

    dataX, dataY = [], []
    for i in range(n_songs):
        lyric = songs[i]
        for j in range(0,len(lyric)-seq_length, 1):
            seq_in = lyric[j:j + seq_length]
            seq_out = lyric[j + seq_length]
            #print(seq_in, seq_out)
            dataX.append([char_to_int[char] for char in seq_in])
            dataY.append(char_to_int[seq_out])
    n_patterns = len(dataX)
    print("Total Patterns: ", n_patterns)

    # prepare
    X = np.reshape(dataX, (n_patterns,seq_length,1))    # reshape X:[samples,time steps,features]
    X = X / float(n_chars)                              # normalize
    y = np_utils.to_categorical(dataY)                  # 1-hot encode the output variable
    print(y.shape, n_chars)

    # save data
    np.save('%sX_sl%d.npy'%(dir_lyrics,seq_length),X)
    np.save('%sy_sl%d.npy'%(dir_lyrics,seq_length),y)
    np.save('%sancillary_sl%d.npy'%dir_lyrics,[char_to_int,int_to_char,n_chars])

if __name__ == '__main__':
    n_songs = -1
    seq_length = [25,50,75,100,125,150,175,200]
    word_or_character = 'character'
    
    #genre = sys.argv[1]
    genre = 'country'

    for sl in seq_length:
        main(genre,n_songs,sl,word_or_character)
