import numpy as np
import sys, glob
from utils.process_lyrics import *
from keras.models import load_model

# text generation
def gen(genre,dir_model,seq_length,word_or_character,embed_dim=50):
    dir_lyrics = 'playlists/%s/'%genre
    
    model = load_model(dir_model)
    text_to_int, int_to_text, len_set = np.load('%sancillary_%s.npy'%(dir_lyrics,word_or_character))
    
    #generate initial seed
    songs = glob.glob('%s/*.txt'%dir_lyrics)
    seed = np.random.randint(0, len(songs))
    ini = list(process_song(songs[seed],word_or_character)[:seq_length])
    pattern = [text_to_int[c] for c in list(ini)]
    print(songs[seed])
    
    # generate text
    if word_or_character == 'character':
        print(''.join([int_to_text[c] for c in pattern]))
        print("****predicted lyrics:****")
        for i in range(300):
            x = np.reshape(pattern, (1, len(pattern), 1))
            x = x / float(len_set)
            pred = model.predict(x, verbose=0)
            #index = np.argmax(pred)
            index = np.random.choice(len(pred[0]), p=pred[0])
            result = int_to_text[index]
            sys.stdout.write(result)
            pattern.append(index)
            pattern = pattern[1:len(pattern)]

    elif word_or_character == 'word':
        print(' '.join([int_to_text[c] for c in pattern]))
        print("****predicted lyrics:****")
        embedding_matrix = np.load('%sembedding_matrix_%dd.npy'%(dir_lyrics,embed_dim))
        matrix_abs = np.abs(embedding_matrix)
        labels = list(text_to_int.keys())
        for i in range(100):
            pred = model.predict(pattern, verbose=0)
            proj = np.sum(pred*embedding_matrix,axis=1)
            index = np.argmax(proj)
            
#            probs = np.nan_to_num(proj/np.sum((np.abs(pred)*matrix_abs),axis=1))
#            probs = (probs - np.min(probs))/(np.max(probs) - np.min(probs))
#            index = np.random.choice(embedding_matrix.shape[0], p=probs)

            result = labels[index]
            sys.stdout.write(result)
            pattern.append(index)
            pattern = pattern[1:len(pattern)]

    print("\nDone.")

if __name__ == '__main__':
    genre = 'pop-rock-edm'
    word_or_character = 'word'
    seq_lengths = [4,6,8,10,12,15]
#    seq_lengths = [25,50,75,100,125,150,175,200]

    for seq_length in seq_lengths:
        print('***********seq_length=%d***********'%seq_length)
        dir_model = 'models/%s_sl%d_%s.h5'%(genre,seq_length,word_or_character)
        gen(genre,dir_model,seq_length,word_or_character)
