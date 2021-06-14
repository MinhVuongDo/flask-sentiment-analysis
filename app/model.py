import tensorflow as tf
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
        
