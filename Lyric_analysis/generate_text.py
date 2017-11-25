import numpy as np
import sys, glob
from keras.models import load_model

# text generation
def gen(genre,dir_model,seq_length):
    dir_lyrics = 'playlists/%s/'%genre
    
    model = load_model(dir_model)
    char_to_int, int_to_char, n_chars = np.load('%sancillary.npy'%dir_lyrics)
    #ini = np.load('%s/X_sl%d.npy'%(dir_lyrics,seq_length))[0]
    songs = glob.glob('playlists/%s/*.txt'%genre)
    seed = np.random.randint(0, n_songs)
    ini = songs[seed][0:seq_length]     #set initial = start of a song
    
    pattern = [int(v*n_chars) for v in list(ini)]
    print(''.join([int_to_char[value] for value in pattern]))
    # generate characters
    for i in range(200):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_chars)
        pred = model.predict(x, verbose=0)
        index = np.argmax(pred)
        #index = np.random.choice(len(pred[0]), p=pred[0])
        result = int_to_char[index]
        #seq_in = [int_to_char[value] for value in pattern]
        sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    print("\nDone.")

if __name__ == '__main__':
    genre = 'country'
    seq_length = 200
    dir_model = 'models/%s_novalid.h5'%genre

    gen(genre,dir_model,seq_length)
