#https://github.com/vlraik/word-level-rnn-keras/blob/master/lstm_text_generation.py

from collections import Counter
import numpy as np
import glob
import sys
from keras.utils import np_utils
from utils.process_lyrics import *

def main(genre,n_songs,seq_length,word_or_character,min_word_occurrence=2):
    
    # get song lyrics
    dir_lyrics = 'playlists/%s/'%genre
    files = glob.glob('%s*.txt'%dir_lyrics)[0:n_songs]
    songs, n_songs = [], len(files)
    for i,f in enumerate(files):
        songs.append(process_song(f, word_or_character))

    # prepare word/character corpus
    if word_or_character == 'character':
        set_ = sorted(list(set(' '.join(songs))))
    elif word_or_character == 'word':
        set_ = []
        for s in songs:
            set_ += s
        set_ = Counter(set_)                    #gets unique sorted dictionary
        for k in list(set_):
            if set_[k] < min_word_occurrence:   #delete rare words from corpus
                del set_[k]
        set_, vals = zip(*set_.most_common())

    # get char/word to int mappings and vice versa.
    len_set = len(set_)      #number of unique words/chars
    text_to_int = dict((c, i) for i, c in enumerate(set_))
    int_to_text = dict((i, c) for i, c in enumerate(set_))
    np.save('%sancillary_%s.npy'%(dir_lyrics,word_or_character),
            [text_to_int,int_to_text,len_set])

    # get data arrays for training LSTMs
    for sl in seq_length:
        dataX, dataY = [], []
        for i in range(n_songs):
            lyric = songs[i]
            for j in range(0,len(lyric)-sl):
                seq_in = lyric[j:j + sl]
                seq_out = lyric[j + sl]
                try:
                    t2i_i = [text_to_int[text] for text in seq_in]
                    t2i_o = text_to_int[seq_out]
                    dataX.append(t2i_i)
                    dataY.append(t2i_o)
                except:
                    # a sparse word->int set is going to yield some
                    # rare words with no matches
                    pass
        n_patterns = len(dataX)
        print("Total Patterns: ", n_patterns)

        # prepare
        X = np.reshape(dataX, (n_patterns,sl,1))        # reshape X:[samples,time steps,features]
        y = np.asarray(dataY)
        if word_or_character == 'character':
            X = X / float(len_set)                      # normalize
            y = np_utils.to_categorical(dataY)          # 1-hot encode the output variable

        # save data (in chunks if too large)
#        x_size, y_size, max_gb = X.nbytes/1e6, y.nbytes/1e6, 4
#        if x_size < max_gb and y_size < max_gb:
        np.save('%sX_sl%d_%s.npy'%(dir_lyrics,sl,word_or_character),X)
        np.save('%sy_sl%d_%s.npy'%(dir_lyrics,sl,word_or_character),y)
#        else:
#            n_chunks = int(round(max(x_size/max_gb, y_size/max_gb)))
#            inc = int(n_patterns/float(n_chunks))
#            print(n_chunks, inc)
#            for i in range(int(n_chunks)):
#                np.save('%sX_sl%d_%s_c%d.npy'%(dir_lyrics,sl,word_or_character,i),
#                        X[inc*i:inc*(i+1)])
#                np.save('%sy_sl%d_%s_c%d.npy'%(dir_lyrics,sl,word_or_character,i),
#                        y[inc*i:inc*(i+1)])


if __name__ == '__main__':
    n_songs = -1
    #seq_length = [25,50,75,100,125,150,175,200]
    seq_length = [6]
    word_or_character = 'word'
    
    #genre = sys.argv[1]
    genre = 'country'
    #genre = 'pop-rock-edm'

    main(genre,n_songs,seq_length,word_or_character)
