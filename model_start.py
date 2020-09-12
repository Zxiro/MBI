import os
import numpy as np
import pandas as pd
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from model_fit import load_data, model_fit
from transformer.transformer import TokenAndPositionEmbedding, TransformerBlock, MultiHeadSelfAttention, load_transformer_model
if "__main__" == __name__:


    if len(sys.argv) < 2:
        stock_symbol = input('enter stock number:')
    else:
        stock_symbol = sys.argv[1]
    day = 5
    x_train, y_train, x_test, y_test, x_val, y_val = load_data(stock_symbol)
    model = load_transformer_model(x_train.shape[-1])
    model_type = "transformer"
    model_fit(model, x_train, y_train, x_test, y_test, x_val, y_val, stock_symbol, model_type)



