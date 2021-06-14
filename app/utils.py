import re
import numpy as np
import os
from model import SentimentAnalysisModel
from underthesea import word_tokenize
from hyper import MAX_SEQ_LENGTH, N_LAYERS, input_shape,LSTM_UNITS,NUM_CLASSES
import tensorflow as tf

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



def load_model(checkpoint_dir):
  model = SentimentAnalysisModel(word_vectors, LSTM_UNITS, N_LAYERS, NUM_CLASSES)
  model.build(input_shape)
  model.load_weights((tf.train.latest_checkpoint(checkpoint_dir)))
  return model