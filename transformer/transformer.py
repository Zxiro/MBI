import os
import numpy as np
import pandas as pd
import sys
#sys.path.append(".")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
#from model_fit import load_data, model_fit
#from ..model_fit import model_fit
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)
        self.config = {
            'embed_dim' : self.embed_dim,
            'num_heads' : self.num_heads,
        }
    def get_config(self):
        '''config = super().get_config().copy()
        config.update({
            'embed_dim' : self.embed_dim,
            'num_heads' : self.num_heads,
            'projection_dim' : self.projection_dim,
            'query_dense' : self.query_dense,
            'key_dense' : self.key_dense,
            'value_dense' : self.value_dense,
            'combine_heads' : self.combine_heads,
        })'''
        return self.config

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.config ={
            'embed_dim':embed_dim,
            'num_heads':num_heads,
            'ff_dim':ff_dim,
            'rate':rate,

        }
    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    def get_config(self):
        '''config = super().get_config().copy()
        config.update({
            'att' : self.att,
            'ffn' : self.ffn,
            'layernorm1' : self.layernorm1,
            'layernorm2' : self.layernorm2,
            'dropout1' : self.dropout1,
            'dropout2' : self.dropout2,
        })'''
        return self.config

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, name=None):
        super(TokenAndPositionEmbedding, self).__init__(name=name)
        #self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.config = {
            "maxlen":maxlen,
            "vocab_size":vocab_size,
            "embed_dim":embed_dim,
        }
    def call(self, x):
        maxlen = tf.shape(x)[-2]
        print(x.shape)
        #[[1,2,3,4,5,2],[]...]
        #[0,1,2,3,4,5]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        #x = self.token_emb(x)
        #[n,maxlen,embed_dim]
        #[n,day,feature]
        return (x + positions)
    def get_config(self):
        '''config = super().get_config().copy()
        config.update({
            'pos_emd': self.pos_emb,
        })
        config ={
            'pos_emd': self.pos_emb,
        }
        base_config = super(TokenAndPositionEmbedding, self).get_config()'''
        return self.config
        '''print(config)
        print(base_config)
        return config'''

def load_transformer_model(embed_dim):
    #embed_dim = x_train.shape[-1]  #Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer

    vocab_size = 20000  # Only consider the top 20k words
    day = 5
    maxlen = day
    inputs = layers.Input(shape=(maxlen,embed_dim))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim, name="TokenAndPositionEmbedding")
    x = embedding_layer(inputs) #[none,5,16]
    #print(x.shape)
    #x = layers.LSTM(16, return_sequences = True, return_state = True)(x)
    #print(x.size)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x) #[none,5,20]
    x = layers.LSTM(100, return_sequences = True)(x)
    x = layers.Dropout(0.1)(x)
    x = layers.LSTM(100, return_sequences = True)(x)
    x = layers.Dropout(0.1)(x)
    x = layers.LSTM(100, return_sequences = False)(x)
    #x = layers.Dense(20, activation="relu")(x) #[none,20] 
    x = layers.Dense(20, activation="tanh")(x) #[none,20]
    x = layers.Dropout(0.2)(x)
    #x = layers.Dense(20, activation="linear")(x) #[none,1]
    '''x = layers.Dense(20)(x)
    x = layers.GlobalAveragePooling1D()(x) #[none,16]
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x) #[none,20]

    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="linear")(x) #[none,1]
    '''
    #print(x.shape)
    #exit()
    #outputs = x
    #model = keras.Model(inputs=inputs,outputs=outputs)
    #print(model.summary())
    outputs = layers.Dense(1)(x)
    #outputs = x
    '''
    index = list(range(len(x_train)))
    np.random.shuffle(index)
    x_train = x_train[index]
    y_train = y_train[index]
    '''
    model = keras.Model(inputs=inputs, outputs=outputs)
    #print(model.summary())
    return model

#if "__main__" == __name__:

    '''
    if len(sys.argv) < 2:
        stock_symbol = input('enter stock number:')
    else:
        stock_symbol = sys.argv[1]
    day = 5
    '''
    '''x_train = np.load('./StockData/TrainingData/NormtrainingX_'+stock_symbol+'.npy')
    y_train = np.load('./StockData/TrainingData/trainingY_'+stock_symbol+'.npy')
    x_test = np.load('./StockData/TrainingData/NormtestingX_'+stock_symbol+'.npy')
    y_test = np.load('./StockData/TrainingData/testingY_'+stock_symbol+'.npy')
    #x_train = np.where(np.isnan(x_train), 0, x_train)
    print(x_train)
    feature = x_train.shape[-1]
    print(feature)
    #exit()
    x_train =x_train.reshape(-1,day,feature)
    x_test = x_test.reshape(-1,day,feature)
    '''
    '''
    x_train, y_train, x_test, y_test, x_val, y_val = load_data(stock_symbol)

    vocab_size = 20000  # Only consider the top 20k words
    '''
    '''maxlen = 200  # Only consider the first 200 words of each movie review
    (x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=vocab_size)
    print(len(x_train), "Training sequences")
    print(len(x_val), "Validation sequences")
    #序列預處理將序列截斷成同樣長度
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)
    '''
    '''
    model = load_model()
    model_type = "transformer"
    model_fit(model, x_train, y_train, x_test, y_test, x_val, y_val, stock_symbol, model_type)
    '''
    '''model.compile("adam", "mse", metrics=["accuracy"])
    callback = EarlyStopping(monitor="val_loss", patience=20, verbose=1, mode="auto")
    history = model.fit(
        x_train, y_train, batch_size=32, epochs=1000, callbacks=[callback], validation_split = 0.15
    )


    #tf.keras.experimental.export_saved_model(model, '../stockModel/transformer_'+stock_symbol+'.h5')
    model.save('./stockModel/transformer_'+stock_symbol+'.h5')
    '''
