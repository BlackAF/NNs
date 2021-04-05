#%%
import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Layer, Input, MaxPooling2D, ReLU, Add, BatchNormalization, GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar100
from tensorflow.data import Dataset

tf.config.list_physical_devices('GPU')

#%%
def load_ds():
    (X_train, y_train), (X_val, y_val) = cifar100.load_data()

    def parse_imgs(imgs, label):
        imgs = tf.cast(imgs, tf.float32)
        imgs = imgs / 255.0
        
        return imgs, label

    train_ds = Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(50000)
    train_ds = train_ds.batch(64)
    train_ds = train_ds.map(parse_imgs, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

    val_ds = Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.shuffle(10000)
    val_ds = val_ds.batch(64)
    val_ds = val_ds.map(parse_imgs, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.experimental.AUTOTUNE)

    return train_ds, val_ds

#%%
class BottleneckBlock(Layer):
    def __init__(self, filters, skip_connection, downsample=False, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = Conv2D(filters, 1, padding='same', strides=2 if downsample else 1)
        self.conv2 = Conv2D(filters, 3, padding='same')
        self.conv3 = Conv2D(filters*4, 1, padding='same')
        self.batch_norm1 = BatchNormalization()
        self.batch_norm2 = BatchNormalization()
        self.batch_norm3 = BatchNormalization()
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
def load_resnet50(input_shape, output_shape):
    resnet50 = Sequential()

    resnet50.add(Input(input_shape))

    resnet50.add(Conv2D(64, 7, strides=2, padding='same', activation='relu'))
    resnet50.add(MaxPooling2D(strides=2))

    # Block 1
    resnet50.add(BottleneckBlock(filters=64, skip_connection='padding'))
    resnet50.add(BottleneckBlock(filters=64, skip_connection='identity'))
    resnet50.add(BottleneckBlock(filters=64, skip_connection='identity'))

    # Block 2
    resnet50.add(BottleneckBlock(filters=128, skip_connection='projection', downsample=True))
    resnet50.add(BottleneckBlock(filters=128, skip_connection='identity'))
    resnet50.add(BottleneckBlock(filters=128, skip_connection='identity'))
    resnet50.add(BottleneckBlock(filters=128, skip_connection='identity'))

    # Block 3
    resnet50.add(BottleneckBlock(filters=256, skip_connection='projection', downsample=True))
    resnet50.add(BottleneckBlock(filters=256, skip_connection='identity'))
    resnet50.add(BottleneckBlock(filters=256, skip_connection='identity'))
    resnet50.add(BottleneckBlock(filters=256, skip_connection='identity'))
    resnet50.add(BottleneckBlock(filters=256, skip_connection='identity'))
    resnet50.add(BottleneckBlock(filters=256, skip_connection='identity'))

    # Block 4
    resnet50.add(BottleneckBlock(filters=512, skip_connection='projection', downsample=True))
    resnet50.add(BottleneckBlock(filters=512, skip_connection='identity'))
    resnet50.add(BottleneckBlock(filters=512, skip_connection='identity'))
    
    # Output
    resnet50.add(GlobalAveragePooling2D())
    resnet50.add(Dense(output_shape, activation='softmax'))

    return resnet50

#%%
resnet50 = load_resnet50(input_shape=(32, 32, 3), output_shape=100)
train_ds, val_ds = load_ds()

resnet50.summary()

resnet50.compile(optimizer=Adam(lr=1e-03), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

epochs = 2
resnet50.fit(train_ds, epochs=epochs, validation_data=val_ds)
