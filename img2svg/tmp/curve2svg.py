#%%
import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt

import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, GlobalAveragePooling2D, Input, Reshape, Layer
from tensorflow.keras.layers import BatchNormalization, ReLU, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.data import Dataset, AUTOTUNE

tf.config.list_physical_devices('GPU')

#%%
class BottleneckBlock(Layer):
    def __init__(self, filters, skip_connection, downsample=False, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = Conv2D(filters, 1, padding='same', strides=2 if downsample else 1)
        self.conv2 = Conv2D(filters, 3, padding='same')
        self.conv3 = Conv2D(filters*4, 1, padding='same')
        self.batch_norm1 = BatchNormalization(momentum=0.9)
        self.batch_norm2 = BatchNormalization(momentum=0.9)
        self.batch_norm3 = BatchNormalization(momentum=0.9)
        self.relu = ReLU()
        self.add = Add()
        self.skip_connection = self.build_skip_connection(skip_connection, filters*4)
        
    def build_skip_connection(self, skip_connection, filters):
        if skip_connection not in ('identity', 'projection', 'padding'):
            raise ValueError('skip_connection must be either identity, projection or padding')
            
        if skip_connection == 'identity':
            return lambda x: x
        
        if skip_connection == 'projection':
            return Conv2D(filters, 1, strides=2)
            
        if skip_connection == 'padding':
            # Pad the last dimension to have the same number of channels once we do the addition
            return lambda x: tf.pad(x, paddings=[[0,0], [0,0], [0,0], [0,abs(filters - x.shape[-1])]])

    def call(self, x, training=False):
        skip_outputs = self.skip_connection(x)

        x = self.conv1(x)
        x = self.batch_norm1(x, training=training)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x, training=training)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.batch_norm3(x, training=training)
        x = self.add([x, skip_outputs])
        x = self.relu(x)

        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'conv1': self.conv1,
            'conv2': self.conv2,
            'conv3': self.conv3,
            'batch_norm1': self.batch_norm1,
            'batch_norm2': self.batch_norm2,
            'batch_norm3': self.batch_norm3,
            'relu': self.relu,
            'add': self.add,
            'skip_connection': self.skip_connection
        })
        
        return config

#%%
class ResNet50(Sequential):
    def __init__(self, input_shape=None, **kwargs):
        super().__init__(**kwargs)
        
        if input_shape:
            self.add(Input(input_shape))

        self.add(Conv2D(64, 7, strides=2, padding='same', activation='relu'))
        self.add(MaxPooling2D(strides=2))

        # Block 1
        self.add(BottleneckBlock(filters=64, skip_connection='padding'))
        self.add(BottleneckBlock(filters=64, skip_connection='identity'))
        self.add(BottleneckBlock(filters=64, skip_connection='identity'))

        # Block 2
        self.add(BottleneckBlock(filters=128, skip_connection='projection', downsample=True))
        self.add(BottleneckBlock(filters=128, skip_connection='identity'))
        self.add(BottleneckBlock(filters=128, skip_connection='identity'))
        self.add(BottleneckBlock(filters=128, skip_connection='identity'))

        # Block 3
        self.add(BottleneckBlock(filters=256, skip_connection='projection', downsample=True))
        self.add(BottleneckBlock(filters=256, skip_connection='identity'))
        self.add(BottleneckBlock(filters=256, skip_connection='identity'))
        self.add(BottleneckBlock(filters=256, skip_connection='identity'))
        self.add(BottleneckBlock(filters=256, skip_connection='identity'))
        self.add(BottleneckBlock(filters=256, skip_connection='identity'))

        # Block 4
        self.add(BottleneckBlock(filters=512, skip_connection='projection', downsample=True))
        self.add(BottleneckBlock(filters=512, skip_connection='identity'))
        self.add(BottleneckBlock(filters=512, skip_connection='identity'))

#%%
class DataManager:
    @classmethod
    def load_dataset(self, log_dir):
        def parse_ds(file_name):
            x_train = tfio.v0.IODataset.from_hdf5(file_name, dataset='/x_train', spec=tf.float32)
            y_train = tfio.v0.IODataset.from_hdf5(file_name, dataset='/y_train', spec=tf.float32)

            return Dataset.zip((x_train, y_train))

        file_names = tf.io.matching_files(os.path.join(log_dir, '*.hdf5'))

        ds = Dataset.from_tensor_slices(file_names)
        ds = ds.shuffle(file_names.shape[0])
        ds = ds.interleave(parse_ds, num_parallel_calls=AUTOTUNE, cycle_length=file_names.shape[0])
        ds = ds.shuffle(10_000)
        ds = ds.repeat(C.EPOCHS)
        ds = ds.batch(C.BATCH_SIZE, drop_remainder=True)
        ds = ds.prefetch(AUTOTUNE)

        return ds

#%%
class TrainManager:
    def __init__(self):
        self.dataset = None
        self.model = None

    def load_dataset(self, refresh=False):
        if self.dataset is not None and refresh == False:
            return

        self.dataset = DataManager.load_dataset(C.DATASET_LOG_DIR)

    def load_model(self, refresh=False):
        if self.model is not None and refresh == False:
            return

        model = Sequential()

        model.add(Input((100, 100, 1)))

        # model.add(ResNet50())
        model.add(Conv2D(64, 3, padding='same', activation='relu'))
        model.add(Conv2D(64, 3, padding='same', activation='relu'))
        model.add(GlobalAveragePooling2D())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(8))
        model.add(Reshape((4, 2)))

        model.compile(Adam(lr=C.INIT_LR), 'mse')

        self.model = model

    def train(self, reload_model=False, reload_data=False):
        self.load_dataset(refresh=reload_data)
        self.load_model(refresh=reload_model)

        return self.model.fit(self.dataset, epochs=C.EPOCHS, steps_per_epoch=C.DATASET_SIZE//C.BATCH_SIZE)

#%%
class C:
    INIT_LR = 1e-03
    DATASET_LOG_DIR = 'datasets'
    DATASET_SIZE = 50_000
    EPOCHS = 1
    BATCH_SIZE = 64


#%%
# tm = TrainManager()

# C.EPOCHS = 50

hist = tm.train(reload_model=False)

plt.plot(hist.history['loss'])

#%%
X, Y = next(iter(tm.dataset))
X = X[0:1]
Y = Y[0:1]

preds = tm.model.predict(X)

print('\n---PREDS---\n', tf.round(preds*10))
print('\n---GROUND---\n', Y*10)

ls = tf.keras.losses.MeanSquaredError()(Y, preds)

print('\n---LOSS---\n', ls.numpy())


