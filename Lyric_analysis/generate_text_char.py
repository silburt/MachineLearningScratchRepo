import numpy as np
import sys, glob
from utils.process_lyrics import *
from keras.models import load_model
import os

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

# text generation
def gen(model, genre, seq_len, temp, song):
    
    dir_lyrics = 'playlists/%s/'%genre
    
    model = load_model(dir_model)
    text_to_int, int_to_text, len_set = np.load('%sancillary_char.npy'%dir_lyrics)
    vocab_size = len(text_to_int)

    # open file and write pred
    song_name = song.split('/')[-1].split('.txt')[0]
    basename = os.path.basename(dir_model).split('h5')[0]
    f = open('playlists/model_predictions/%s/%s_temp%.2f.txt'%(genre, basename, temp), 'a')
    f.write("\n********** Song is %s ********** \n"%song_name)
    #f = open('playlists/model_predictions/%s/%s_sl%s_n2_temp%.2f.txt'%(genre, name, seq_len, temp), 'w')
    #f.write("Model used: %s\n\n"%dir_model)

    # generate text
    lyrics = process_song(song)
    #n_chars = len(lyrics)
    n_chars = 600
    f.write(lyrics[:seq_len])
    pattern = [text_to_int[c] for c in list(lyrics[:seq_len])]
    print(''.join([int_to_text[c] for c in pattern]))
    print("****predicted lyrics for sl=%d, temp=%f:****"%(seq_len, temp))
    i, result = 0, ''
    while True:
        x = np.eye(vocab_size)[pattern].reshape(1,seq_len, vocab_size)
        preds = model.predict(x, verbose=0)
        pred = preds.reshape(seq_len, vocab_size)[-1]
        
        # sample
        index = sample(pred, temp)
        result = int_to_text[index]
        f.write(result)
        
        # update pattern
        sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

        # break sequence
        if (i >= n_chars) and (result == '\n'):
            break
        i += 1
    print("\nDone.")
    f.close()

if __name__ == '__main__':
    n_songs = 1
    genre = 'country'
    seq_length = 150
    temperatures = [0.1,0.2,0.4]
    #dir_model = 'models/pop-rock-edm_sl150_nl1_size1024_bs256_drop0.0.h5' #temp=0.4 is nice
    #dir_models = glob.glob('models/edm_*.h5')
    dir_models = ['models/country_sl150_nl1_size64_bs256_drop0.2.h5']

    songs = glob.glob('playlists/%s/*.txt'%genre)
    for dir_model in dir_models:
        for i in range(n_songs):
            for temp in temperatures:
                gen(dir_model, genre, seq_length, temp, songs[i])
