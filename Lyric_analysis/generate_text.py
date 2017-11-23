import numpy as np
import glob
import sys
from keras.models import load_model
from clean_lyrics import *

# text generation
def gen(genre,dir_model,seq_length):
    dir_lyrics = 'playlists/%s/'%genre
    X = np.load('%s/X_sl%d.npy'%(dir_lyrics,seq_length))
    char_to_int = np.load('%schar_to_int.npy'%dir_lyrics)
    int_to_char = np.load('%sint_to_char.npy'%dir_lyrics)
    
    #seed = np.random.randint(0, n_songs)
    #ini = songs[seed][0:seq_length]     #set initial = start of a song
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
    genre = 'country'
    seq_length = 200
    dir_model = 'models/%s.h5'%genre

    gen(genre,dir_model,seq_length)
