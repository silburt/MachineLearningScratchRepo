import numpy as np
np.random.seed(2)

genre = 'country'
word_or_character = 'character'
N_check = 50
embed_dim = 50
#seq_length = [4]#,6,8,10,15]
seq_length = [125]

dir = 'playlists/%s/'%genre
text_to_int, int_to_text, n_chars = np.load('%sancillary_%s.npy'%(dir,word_or_character))
if word_or_character == 'word':
    em = np.load('%sembedding_matrix_%dd.npy'%(dir,embed_dim))
    em_norms = np.sqrt(np.sum(em*em,axis=1))
    em_norms[em_norms==0] = -1
    labels = list(text_to_int.keys())

err_count = 0
for sl in seq_length:
    X = np.load('%sX_sl%d_%s.npy'%(dir,sl,word_or_character))
    y = np.load('%sy_sl%d_%s.npy'%(dir,sl,word_or_character))
    y_raw = np.load('%syraw_sl%d_%s.npy'%(dir,sl,word_or_character))
    song_names = np.load('%ssong_names_sl%d_%s.npy'%(dir,sl,word_or_character))

    ran = np.random.randint(0,X.shape[0],N_check)
    for r in ran:
        ex = []
        for x in X[r]:
            ex.append(int_to_text[x])
        if word_or_character == 'word':
            y_norm = np.sqrt(np.sum(y[r]*y[r]))
            norm = np.sum(y[r]*em,axis=1)/(em_norms*y_norm)
            argmax = np.argmax(norm)
            #argmax = np.argmax(np.sum(y[r]*em,axis=1))
            why = labels[argmax]
            if why != int_to_text[y_raw[r]]:
                print(int_to_text[y_raw[r]])
                print('true',norm[y_raw[r]])
                print('result',norm[argmax])
                err_count += 1
        elif word_or_character == 'character':
            why = int_to_text[y[r]]
        print(ex, why, '--', song_names[r])
        print('\n')

print('err_count = %d'%err_count)
