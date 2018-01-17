import numpy as np
import sys, glob
from utils.process_lyrics import *
from keras.models import load_model

# From https://groups.google.com/forum/#!msg/keras-users/Y_FG_YEkjXs/nSLTa2JK2VoJ
# Francois Chollet: "It turns out that the 'temperature' for sampling (or more
# generally the choice of the sampling strategy) is critical to get sensible results.
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# generate initial sequence
def init_sequence(genre, seq_length):
    songs = glob.glob('playlists/%s/*.txt'%genre)
    seed = np.random.randint(0, len(songs))
    return list(process_song(songs[seed])[:seq_length])

# text generation
def gen(genre, seq_length, temperature, ini):
    dir_lyrics = 'playlists/%s/'%genre
    dir_model = 'models/%s_sl%d_char.h5'%(genre, seq_length)
    
    model = load_model(dir_model)
    text_to_int, int_to_text, len_set = np.load('%sancillary_char.npy'%dir_lyrics)
    vocab_size = len(text_to_int)

    # generate text
    pattern = [text_to_int[c] for c in list(ini)]
    print(''.join([int_to_text[c] for c in pattern]))
    print("****predicted lyrics for sl=%d, temp=%f:****"%(seq_length,temperature))
    for i in range(300):
        x = np.eye(vocab_size)[pattern].reshape(1,seq_length,vocab_size)
        preds = model.predict(x, verbose=0)
        pred = preds.reshape(seq_length,vocab_size)[-1]
        
        # sample
        index = sample(pred, temperature)
        result = int_to_text[index]
        
        # update pattern
        sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    print("\nDone.")

if __name__ == '__main__':
    genre = 'pop-rock-edm'
    temperatures = [0.2,0.5,1.0,1.2]
    seq_lengths = [150]
    #seq_lengths = [25,50,75,100,125,150,175,200]

    for seq_length in seq_lengths:
        ini_seq = init_sequence(genre, seq_length)
        for temp in temperatures:
            gen(genre, seq_length, temp, ini_seq)
