import numpy as np

genre = 'pop-rock-edm'
word_or_character = 'word'
N_check = 50
embed_dim = 50
seq_length = [4]#,6,8,10,15]

dir = 'playlists/%s/'%genre
text_to_int, int_to_text, n_chars = np.load('%sancillary_%s.npy'%(dir,word_or_character))
if word_or_character == 'word':
    embedding_matrix = np.load('%sembedding_matrix_%dd.npy'%(dir,embed_dim))
    labels = list(text_to_int.keys())

for sl in seq_length:
    X = np.load('%sX_sl%d_%s.npy'%(dir,sl,word_or_character))
    y = np.load('%sy_sl%d_%s.npy'%(dir,sl,word_or_character))
    song_names = np.load('%ssong_names_sl%d_%s.npy'%(dir,sl,word_or_character))

    ran = np.random.randint(0,X.shape[0],N_check)
    for r in ran:
        ex = []
        for x in X[r]:
            ex.append(int_to_text[x])
        if word_or_character == 'word':
            why = labels[np.argmax(np.sum(y[r]*embedding_matrix,axis=1))]
        elif word_or_character == 'character':
            why = int_to_text[y[r]]
        print(ex, why, song_names[r])
