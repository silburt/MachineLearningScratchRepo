import numpy as np
import sys
from keras.models import load_model

# text generation
def gen(genre,dir_model,seq_length):
    dir_lyrics = 'playlists/%s/'%genre
    
    model = load_model(dir_model)
    X = np.load('%s/X_sl%d_2.npy'%(dir_lyrics,seq_length))
    char_to_int, int_to_char, n_chars = np.load('%sancilary_2.npy'%dir_lyrics)
    
    #seed = np.random.randint(0, n_songs)
    #ini = songs[seed][0:seq_length]     #set initial = start of a song
    ini = X[0]
    pattern = [int(v*n_chars) for v in list(ini)]
    print([int_to_char[value] for value in pattern])
    # generate characters
    for i in range(200):
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
    seq_length = 200
    dir_model = 'models/%s.h5'%genre

    gen(genre,dir_model,seq_length)
