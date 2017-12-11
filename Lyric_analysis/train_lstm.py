#https://danijar.com/tips-for-training-recurrent-neural-networks/
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Embedding
#from keras.layers import Lambda, Input
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
from sklearn.model_selection import train_test_split
import tensorflow as tf
import sys

# https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
def get_embedding_matrix(text_to_int,embed_dim,y):
    f = open('utils/glove.6B/glove.6B.%dd.txt'%embed_dim,'r')
    
    embeddings_index = {}
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((len(text_to_int), embed_dim))
    for word, i in text_to_int.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # convert y-data to embed
    for 

    return embedding_matrix

def train_model(genre,dir_model,seq_length,epochs,batch_size,word_or_character,embed_dim=50):
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True)) #check gpu is being used
    
    X = np.load('playlists/%s/X_sl%d_%s.npy'%(genre,seq_length,word_or_character))
    y = np.load('playlists/%s/y_sl%d_%s.npy'%(genre,seq_length,word_or_character))
    
    if word_or_character == 'word':
        embedding_matrix, y = get_embedding_matrix(text_to_int,embed_dim,y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    text_to_int, int_to_text, n_chars = np.load('%sancillary_%s.npy'%%(dir_lyrics,word_or_character))
    
    try:
        model = load_model(dir_model)
        print("successfully loaded previous model, continuing to train")
    except:
        print("generating new model")
        model = Sequential()
        
        nb_classes = len(char_to_int)
        #input_shape, output_shape = (seq_length,), (input_shape[0], nb_classes)
        #input = Input(shape=input_shape, dtype='uint8')
        #model.add(Lambda(K.one_hot,arguments={'nb_classes': nb_classes}, output_shape=output_shape)(input))
        
        # old network
#        model.add(LSTM(512, dropout=0.2, recurrent_dropout=0.2, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
#        model.add(LSTM(512, dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
#        model.add(Dense(y.shape[1], activation='softmax'))
#        model.add(GRU(512, dropout=0.2, recurrent_dropout=0.2, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
#        model.add(GRU(512, dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
#        model.add(Dense(y.shape[1], activation='softmax'))

        #embedding- so probably the thing to do is use a frozen weight pre-trained embedding layer (word2vec), and the output is an embedding vector. I.e. convert all your words to embedding.
        model.add(Embedding(len(text_to_int), embed_dim, weights=[embedding_matrix],
                            input_length=seq_length, trainable=False))
        model.add(GRU(embed_dim, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
        model.add(GRU(512, dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
        model.add(Dense(len(text_to_int)))

        optimizer = Adam(lr=2e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    print(model.summary())
    checkpoint = ModelCheckpoint(dir_model, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
              callbacks=callbacks_list, validation_data=(X_test, y_test), verbose=2)
    model.save(dir_model)

if __name__ == '__main__':
    genre = 'country'
    word_or_character = 'word'
    #seq_length = 200
    seq_length = int(sys.argv[1])
    dir_model = 'models/%s_sl%d_%s.h5'%(genre,seq_length,word_or_character)
    
    epochs = 40
    batch_size = 256
    
    train_model(genre,dir_model,seq_length,epochs,batch_size,word_or_character)



