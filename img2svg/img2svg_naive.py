#%%
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import sys

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Layer, BatchNormalization, ReLU, Add
from tensorflow.keras.layers import GRU, TimeDistributed, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.data import Dataset, TextLineDataset
from tensorflow.data.experimental import CsvDataset

# sys.stdout = open('/dev/stdout', 'w')

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
#         x = self.batch_norm1(x, training=training)
        x = self.relu(x)
        x = self.conv2(x)
#         x = self.batch_norm2(x, training=training)
        x = self.relu(x)
        x = self.conv3(x)
#         x = self.batch_norm3(x, training=training)
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
class C:
    BATCH_SIZE = 2

#%%
class DataManager:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.MASK_TOKEN = -1.0
        self.START_TOKEN = -2.0
        self.END_TOKEN = -3.0

    def load_dataset(self):
        def build_ds(set_name):
            ds = CsvDataset(str(pathlib.Path(self.log_dir, set_name, 'dataset.csv')), record_defaults=['',''])
            # ds = ds.shuffle(25000)
            ds = ds.take(C.BATCH_SIZE*1)
            ds = ds.batch(C.BATCH_SIZE, drop_remainder=True)
            ds = ds.map(lambda *x: self.parse_batch(*x, set_name), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
            return ds
        
        train_ds = build_ds('train')
        val_ds = build_ds('validation')

        return train_ds, val_ds
    
    def parse_batch(self, file_names, labels, set_name):
        imgs = self.parse_imgs(file_names, set_name)
        input_labels, target_labels = self.parse_labels(labels)

        return (imgs, input_labels), target_labels
        
    def parse_imgs(self, file_names, set_name):
        def parse_img(file_name, set_name):
            img_path = tf.strings.join([self.log_dir, f'/{set_name}/imgs/', file_name, '.png'])

            img = tf.io.read_file(img_path)
            img = tf.io.decode_png(img, channels=3)
            img = tf.cast(img, tf.float32)
            img = img / 255.0

            return img

        # The loading of images can't be vectorized so map over it, parallel execution is enabled by default (10)
        imgs = tf.map_fn(lambda x: parse_img(x, set_name), file_names, fn_output_signature=tf.float32)
        
        return imgs
    
    def parse_labels(self, labels):
        labels = [f'{self.START_TOKEN}', labels, f'{self.END_TOKEN}']
        labels = tf.strings.join(labels, separator=',')
        labels = tf.strings.split(labels, sep=',')
        labels = tf.strings.to_number(labels, out_type=tf.float32)
        
        # Convert from ragged_tensor to tensor, it will be padded using the mask token
        labels = labels.to_tensor(self.MASK_TOKEN)

        # The input exludes the end token
        input_labels = labels[..., :-1]
        # The target excludes the start token
        target_labels = labels[..., 1:]

        # RNNs expects a 3D input, so convert (batch,time_steps) to (batch,time_steps,1)
        input_labels = tf.expand_dims(input_labels, -1)
        target_labels = tf.expand_dims(target_labels, -1)
        
        return input_labels, target_labels
    
# dm = DataManager('datasets/mixed')
# ds, _ = dm.load_dataset()

# for _ in ds:
#     print(_)
#     print()


#%%
class IMG2SVG(Model):
    def __init__(self, mask_token, **kwargs):
        super().__init__(**kwargs)
        self.mask_token = mask_token

        self.resnet50 = ResNet50()
        self.global_avg_pool = GlobalAveragePooling2D()
        self.gru = GRU(2048, return_sequences=True)
        self.time_distributed1 = TimeDistributed(Dense(128, activation='relu'))
#         self.time_distributed2 = Dense(64, activation='relu')
#         self.time_distributed3 = TimeDistributed(Dense(32, activation='relu'))
#         self.time_distributed4 = TimeDistributed(Dense(16, activation='relu'))
        self.time_distributed5 = TimeDistributed(Dense(1))
        self.masking = Masking(mask_token)
        
    def call(self, data, training=False):
        imgs, input_labels = data
        
        # Encode the images
        imgs = self.resnet50(imgs, training=training)
        imgs = self.global_avg_pool(imgs)
        
        # Use the encoded vectors of the images as initial state
        initial_state = imgs
        
        # Make predictions
        outputs = self.masking(input_labels)
        outputs = self.gru(outputs, initial_state=initial_state, training=training)
        outputs = self.time_distributed1(outputs)
#         outputs = self.time_distributed2(outputs)
#         outputs = self.time_distributed3(outputs)
#         outputs = self.time_distributed4(outputs)
        outputs = self.time_distributed5(outputs)
        
        return outputs

#%%
# i = IMG2SVG()

# lbl = tf.ragged.constant([
#     [-1, 3, 4],
#     [-1, 5, 6]
# ], dtype=tf.float32)
# lbl = tf.expand_dims(lbl, axis=-1)
# outputs = i({ 'imgs': tf.zeros((2,100,100,3)), 'input_labels': lbl })

# print('outputs')
# print(outputs.shape)

#%%
dm = DataManager('datasets/mixed')
train_ds, val_ds = dm.load_dataset()

#%%
model = IMG2SVG(mask_token=dm.MASK_TOKEN)
model.compile(Adam(lr=1e-04), loss='mse')

#%%
hist = model.fit(train_ds, epochs=1)

#%%
plt.subplot(1, 2, 1)
plt.plot(hist.history['loss'], label='loss')
# plt.plot(hist.history['val_loss'], label='val_loss')
plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(hist.history['acc'], label='acc')
# plt.plot(hist.history['val_acc'], label='val_acc')
# plt.legend()
# plt.tight_layout()
plt.show()

#%%
x,y = next(iter(train_ds))
# x = x[:1]
# y = y[:1]

preds = model.predict(x, batch_size=1)
# preds = tf.squeeze(preds)

print('y         \n', y.numpy())
print()
print('preds      \n', preds)
print()
print('sub        \n', (y-preds).numpy())
print()
print('subsquared \n', ((y-preds)**2).numpy())
print()
print('sum        \n', tf.math.reduce_sum((y-preds)**2).numpy())
print()
print('avg        \n', tf.math.reduce_sum((y-preds)**2).numpy() / 4.0)
print()
print('loss       \n', tf.keras.losses.MeanSquaredError()(y, preds).numpy())
