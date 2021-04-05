#%%
import tensorflow as tf
import numpy as np
import string
import pathlib
import os

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import GRU, Embedding, Input, Dense, GlobalAveragePooling2D, Add
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.optimizers import Adam
from tensorflow.io.gfile import GFile
from tensorflow.strings import unicode_split
from tensorflow.data import Dataset, TextLineDataset

#%%
class IMG2SVG(Model):
    def __init__(self, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size

        self.img_encoder = self.load_img_encoder()
        self.svg_encoder = self.load_svg_encoder()
        self.add = Add()
        self.dense = Dense(vocab_size, activation='softmax')

    def load_img_encoder(self):
        img_encoder = Sequential()

        img_encoder.add(EfficientNetB1(input_shape=(100,100,3), include_top=False, weights=None))
        img_encoder.add(GlobalAveragePooling2D())

        return img_encoder
    
    def load_svg_encoder(self):
        svg_encoder = Sequential()

        svg_encoder.add(Embedding(input_dim=self.vocab_size, output_dim=C.EMBEDDING_SIZE))
        svg_encoder.add(GRU(1280, return_sequences=True))

        return svg_encoder
    
    def call(self, x, training=False):
        svg, img = x

        img_input = self.img_encoder(img, training=training)
        svg_input = self.svg_encoder(svg, training=training)

        x = self.add([img_input, svg_input])
        x = self.dense(x)
        
        return x


    def train_step(self, data):
        x, y = data
        
        with tf.GradientTape() as tape:
            preds = self(x, training=True) # (BATCH,SEQUENCE_LEN,VOCAB_SIZE) 
            # Choose the highest probability for each char
            preds = tf.math.reduce_max(preds, axis=-1) # (BATCH,SEQUENCE_LEN)
            loss = self.compiled_loss(y, preds)
        
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        return { m.name: m.result() for m in self.metrics }
        
img2svg = IMG2SVG(vocab_size=dm.vocab_size)

img2svg.compile(Adam(lr=1e-03), loss='mse')

img2svg.fit(ds, epochs=20)

#%%
class DataManager:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.START_TOKEN = '[SOS]'
        self.END_TOKEN = '[EOS]'
        self.vocab = list(sorted(set(string.printable))) + [self.START_TOKEN, self.END_TOKEN]
        self.chars_to_ids = StringLookup(vocabulary=self.vocab)
        self.vocab_size = self.chars_to_ids.vocab_size()

    def load_dataset(self):
        ds = TextLineDataset(str(pathlib.Path(self.log_dir, 'file_names.txt')))
        ds = ds.take(5)
        ds = ds.map(self.parse_svg_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.padded_batch(2, drop_remainder=True)
        
        return ds
            
    def parse_svg_img(self, file_name):
        svg_path = tf.strings.join([self.log_dir, '/svgs/', file_name, '.svg'])
        img_path = tf.strings.join([self.log_dir, '/imgs/', file_name, '.png'])

        svg = tf.io.read_file(svg_path)
        svg = tf.concat([[self.START_TOKEN], unicode_split(svg, 'UTF-8'), [self.END_TOKEN]], axis=0)
        svg = self.chars_to_ids(svg)
        
        img = tf.io.read_file(img_path)
        img = tf.io.decode_png(img, channels=3)
        img = tf.cast(img, tf.float32)
        img = img / 255.0
        
        return (svg, img), svg

dm = DataManager('dataset')

ds = dm.load_dataset()

for _,__ in ds:
    print(_[0].shape, _[1].shape, __.shape)

#%%
class C:
    EMBEDDING_SIZE = 256
