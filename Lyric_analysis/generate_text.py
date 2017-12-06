import numpy as np
import sys, glob
from keras.models import load_model
from process_lyrics import *

# text generation
def gen(genre,dir_model,seq_length):
    dir_lyrics = 'playlists/%s/'%genre
    
    model = load_model(dir_model)
    char_to_int, int_to_char, n_chars = np.load('%sancillary_sl%d.npy'%(dir_lyrics,seq_length))
    
    #generate initial seed
    #ini = np.load('%s/X_sl%d.npy'%(dir_lyrics,seq_length))[0]
    #pattern = [int(c*n_chars) for c in list(ini)]
    
    #generate initial seed
    songs = glob.glob('%s/*.txt'%dir_lyrics)
    seed = np.random.randint(0, len(songs))
    ini = list(process_song(songs[seed],'character')[:seq_length])
    pattern = [char_to_int[c] for c in list(ini)]
    print(songs[seed])
    print(''.join([int_to_char[c] for c in pattern]))
    print("*****")
    
    # generate characters
    for i in range(300):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_chars)
        pred = model.predict(x, verbose=0)
        #index = np.argmax(pred)
        index = np.random.choice(len(pred[0]), p=pred[0])
        result = int_to_char[index]
        #seq_in = [int_to_char[value] for value in pattern]
        sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    print("\nDone.")

if __name__ == '__main__':
    genre = 'country'
    seq_lengths = [25,50,75,100,125,150,175,200]
    
    for seq_length in seq_lengths:
        print('***********seq_length=%d***********'%seq_length)
        dir_model = 'models/%s_novalid.h5'%genre
        gen(genre,dir_model,seq_length)
