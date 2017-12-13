import numpy as np

genre = 'pop-rock-edm'
word_or_character = 'word'
N_check = 50
embed_dim = 50
seq_length = [4]#,6,8,10,15]

text_to_int, int_to_text, n_chars = np.load('playlists/%s/ancillary_%s.npy'%(genre,word_or_character))
if word_or_character == 'word':
    embedding_matrix = np.load('playlists/%s/embedding_matrix_%dd.npy'%(genre,embed_dim))
    labels = list(text_to_int.keys())

for sl in seq_length:
    X = np.load('playlists/%s/X_sl%d_%s.npy'%(genre,sl,word_or_character))
    y = np.load('playlists/%s/y_sl%d_%s.npy'%(genre,sl,word_or_character))

    ran = np.random.randint(0,X.shape[0],N_check)
    for r in ran:
        ex = []
        for x in X[r]:
            ex.append(int_to_text[x])
        if word_or_character == 'word':
            why = labels[np.argmax(np.sum(y[r]*embedding_matrix,axis=1))]
        elif word_or_character == 'character':
            why = int_to_text[y[r]]
        print(ex, why)
