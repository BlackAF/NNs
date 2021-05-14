#%%
import tensorflow as tf

from tensorflow.keras.layers import Layer

# %%
class Attention(Layer):
    def __init__(self, keys, values, queries, **kwargs):
        super().__init__(**kwargs)

    def call(self):
        dk = 4
        db = 4
        keys = Dense(dk)(keys)
        values = Dense(dv)(values)
        queries = Dense(dk)(queries)

        qk = Dot(axis=1)(queries, keys)
        scaled = Lambda(lambda x: x / (dk ** 1/2))(qk)
        sm = Softmax()(scaled)

        out = Dot(axis=1)(sm, values)
        out = Concat()([out])
        out = Dense(?)(out)