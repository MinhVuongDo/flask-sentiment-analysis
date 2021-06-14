import re
import numpy as np
import os
# from model import SentimentAnalysisModel
from underthesea import word_tokenize
# from hyper import MAX_SEQ_LENGTH, N_LAYERS, input_shape
import tensorflow as tf
MAX_SEQ_LENGTH = 200
data_dir ='./vocab/'
words_list = np.load(os.path.join(data_dir, 'words_list.npy')).tolist()

word_vectors = np.float32(np.load(os.path.join(data_dir, 'word_vectors.npy')))



word2idx = {w:i for i,w in enumerate(words_list)}

word2idx['UNK']

def get_sentence_indices(sentence, max_seq_length, _words_list):
    
    indices = np.zeros((max_seq_length), dtype='int32')
    
    words = [word.lower() for word in sentence.split()]

    unk_idx = word2idx['UNK']
    
    
    for idx, word in enumerate(words):
      if idx >= len(indices):
        break
      if word in _words_list:
        indices[idx] = word2idx[word]
        # indices[idx] = _words_list.index(word)
      else:
        indices[idx] = unk_idx

    return indices

def clean_sentences(string):
    strip_special_chars = re.compile("[^\w0-9 ]+")
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())


def predict(sentence, model, _word_list=words_list, _max_seq_length=MAX_SEQ_LENGTH):

    tokenized_sent = word_tokenize(sentence, format='text')
    
    input_data = [get_sentence_indices(clean_sentences(tokenized_sent),
                                        MAX_SEQ_LENGTH,words_list)]
   
    predictions = np.argmax(model(input_data),1)[0]

    return predictions


lstm_layer = tf.keras.layers.LSTM

class SentimentAnalysisModel(tf.keras.Model):
    
    def __init__(self, word2vec, lstm_units, n_layers, num_classes, dropout_rate=0.25):
        
        super().__init__(name='sentiment_analysis')
        
        
        self.word2vec = word2vec
        
        self.lstm_layers = []  # List chứa các tầng LSTM
        self.dropout_layers = []  # List chứa các tầng dropout

        for i in range(n_layers):
            new_lstm = lstm_layer(lstm_units, name='lstm_'+str(i), return_sequences=True if i<n_layers-1 else False)
            # new_lstm = lstm_layer(lstm_units, return_sequences=True)
            self.lstm_layers.append(new_lstm)
            new_dropout = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout_'+str(i))
            self.dropout_layers.append(new_dropout)
        
        self.dense_layer = tf.keras.layers.Dense(units=num_classes, activation='softmax', name='dense_0')
        ### END CODE HERE
        
    def call(self, inputs):
        
        inputs = tf.cast(inputs, tf.int32)
        
        x = tf.nn.embedding_lookup(self.word2vec, inputs)
        
        for i in range(len(self.lstm_layers)):
            x = self.lstm_layers[i](x)
            x = self.dropout_layers[i](x)
     
        
        out = self.dense_layer(x)
        
        return out
model = SentimentAnalysisModel(word_vectors, lstm_units = 256, n_layers = 2, num_classes = 2)
model.load_weights(tf.train.latest_checkpoint('../model/'))
model.build(input_shape=(1,200))
model.summary()



predict('mot hai ba', model)