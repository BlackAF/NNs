#%%
import logging
logging.disable(logging.WARNING)

import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import glob
import json
import pathlib
import io
import h5py
import math

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input
from tensorflow.keras.models import Sequential, Model, clone_model
from tensorflow.keras.layers import Dense, Input, Layer, Flatten, Lambda, UpSampling2D, Conv2D, Dropout
from tensorflow.keras.losses import Loss, MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
from tensorflow.keras.callbacks import Callback, TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.constraints import MinMaxNorm
from tensorflow.data import Dataset
from tensorflow.io.gfile import GFile

tf.get_logger().setLevel('ERROR')
tf.config.list_physical_devices('GPU')

#%%
BATCH_SIZE = 1

# Layers to take from VGG19
STYLE_LAYERS = ('block1_conv1', 'block1_conv2', 'block1_pool', 'block2_conv1', 'block2_conv2', 'block2_pool', 'block3_conv1', 'block3_conv2',
                'block3_conv3', 'block3_conv4', 'block3_pool')

DECODER_LAYERS = ('block3_pool', 'block3_conv1', 'block2_pool', 'block2_conv1', 'block1_pool', 'block1_conv1')

#%%
def get_dataset():
    def parse_img(file_name):
        img = tf.io.read_file(file_name)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [224, 224], antialias=True, method='nearest')
        img = tf.cast(img, tf.float32)
        img = vgg19_preprocess_input(img) / 255.0
        
        return img

    def build_ds(file_names):
        tmp_ds = Dataset.from_tensor_slices(file_names)
        tmp_ds = tmp_ds.shuffle(len(file_names))
        tmp_ds = tmp_ds.map(parse_img, num_parallel_calls=tf.data.AUTOTUNE)
        
        return tmp_ds
    
    content_ds = build_ds(glob.glob('/home/jephthia/datasets/mscoco/unlabeled2017/train/*')[:1])
    # style_ds = build_ds(glob.glob('/home/jephthia/datasets/wikiart/train/*')[:1])
    
    ds = content_ds.batch(BATCH_SIZE)
   
    return ds

#%%
vgg19 = VGG19()

#%%
class SaveModel(Callback):
    def __init__(self, log_dir, update_freq=1, **kwargs):
        super(SaveModel, self).__init__(**kwargs)
        self.log_dir = log_dir
        self.update_freq = update_freq
        
    def on_epoch_end(self, epoch, logs):
        epoch = epoch + 1
        if epoch % self.update_freq == 0:
            self.model.save_model(log_dir=os.path.join(self.log_dir, f'epoch_{epoch}'))

#%%
class Conv2DReflectivePadding(Layer):
    def __init__(self, *args, conv2d=None, **kwargs):
        super().__init__(name=kwargs.get('name', None))
        
        if conv2d is None:
            self.conv2d = Conv2D(*args, **kwargs)
        else:
            self.conv2d = conv2d

    def call(self, x):
        x = tf.pad(x, tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]), 'REFLECT')
        x = self.conv2d(x)

        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({'conv2d': self.conv2d})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(conv2d=config['conv2d'])

#%%
class StyleTransfer(Model):
    def __init__(self, *args, encoder=None, decoder=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_encoder(encoder)
        self.init_decoder(decoder)
        self.build_metrics()
    
    def init_encoder(self, encoder):
        if encoder is not None:
            self.encoder = encoder
            return

        self.encoder = Sequential([Input((224,224,3))])
        self.encoder.trainable = False

        for layer_name in STYLE_LAYERS:
            layer = vgg19.get_layer(layer_name)
            self.encoder.add(layer)
            
        self.encoder.compile()
        
    def init_decoder(self, decoder):
        if decoder is not None:
            self.decoder = decoder
            return

        # Build the decoder if it wasn't provided
        input_shape = self.encoder.layers[-1].output_shape[1:]
        self.decoder = Sequential([Input(input_shape)])

        # The decoder is the trimed inverse of the encoder
        for layer_name in DECODER_LAYERS:
            layer = vgg19.get_layer(layer_name)
            # Add the upsampling to double the image size                
            if 'pool' in layer.name:
                block_name = layer.name.split("_")[0]
                self.decoder.add(UpSampling2D(name=f'{block_name}_upsampling'))
            # Add some reflective padding followed by a Conv2D layer
            elif 'conv' in layer.name:
                self.decoder.add(Conv2DReflectivePadding(
                    filters=layer.output_shape[-1],
                    kernel_size=layer.kernel_size,
                    strides=layer.strides,
                    activation='relu',
                    name=layer.name))
                
        # Add one final Conv2D to reduce the feature maps to 3 (N,W,H,3)
        self.decoder.add(Conv2DReflectivePadding(3, (3,3), name='output_conv1'))
    
    def build_metrics(self):
        self.c_loss_metric = Mean(name='c_loss')

    def compile(self, optimizer, content_loss, **kwargs):
        super().compile(**kwargs)
        if not getattr(self, 'decoder_compiled', False):
             self.decoder.compile(optimizer=optimizer)
        self.content_loss = content_loss
    
    @tf.function
    def train_step(self, data, training=True):
        c_encoded_outputs = data
            
        with tf.GradientTape(watch_accessed_variables=training) as tape:
            # 1. Encode the content and style image
            for layer in self.encoder.layers:
                # Encode the content image
                c_encoded_outputs = layer(c_encoded_outputs)

            # 3. Decode the feature maps generated by AdaIN to get the final generated image
            generated_imgs = self.decoder(c_encoded_outputs, training=training)

            # 4. Encode the generated image to calculate the loss
            g_encoded_outputs = generated_imgs
            for layer in self.encoder.layers:
                # Encode the generated image
                g_encoded_outputs = layer(g_encoded_outputs)
                    
            # 5. Calculate the content loss
            c_per_replica_loss = self.content_loss(g_encoded_outputs, c_encoded_outputs) # (N,W,H)
            # Reduce the loss (we do this ourselves in order to be compatible with distributed training)
            global_c_loss_size = tf.size(c_per_replica_loss) * self.distribute_strategy.num_replicas_in_sync
            global_c_loss_size = tf.cast(global_c_loss_size, dtype=tf.float32)
            c_loss = tf.nn.compute_average_loss(c_per_replica_loss, global_batch_size=global_c_loss_size)

        # 8. Apply gradient descent
        if training:
            gradients = tape.gradient(c_loss, self.decoder.trainable_variables)
#             gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
#             tf.print('---')
#             tf.print('glonorm', tf.linalg.global_norm(gradients))
#             tf.print(list((i, tf.math.reduce_min(n), tf.math.reduce_max(n)) for i,n in enumerate(gradients)))
#             tf.print(list((i, tf.math.reduce_min(n), tf.math.reduce_max(n)) for i,n in enumerate(self.decoder.trainable_variables)))
#             tf.print('C_ENC --> ', tf.math.reduce_min(c_encoded_outputs), tf.math.reduce_max(c_encoded_outputs))
#             tf.print('GEN_IMG --> ', tf.math.reduce_min(generated_imgs), tf.math.reduce_max(generated_imgs))
#             tf.print('G_ENC --> ', tf.math.reduce_min(g_encoded_outputs), tf.math.reduce_max(g_encoded_outputs))

            self.decoder.optimizer.apply_gradients(zip(gradients, self.decoder.trainable_variables))
        
        # 9. Update the metrics
        self.c_loss_metric.update_state(c_loss)

        return { m.name: m.result() for m in self.metrics }

    @tf.function
    def predict_step(self, data):
        content_imgs = data
        
        # Ensure these are batched
        assert len(content_imgs.shape) == 4

        content_imgs = vgg19_preprocess_input(content_imgs) / 255.0

        c_encoded = content_imgs

        # Encode the contents and styles
        for layer in self.encoder.layers:
            c_encoded = layer(c_encoded)
        
        # Decode the images to generate them
        generated_imgs = self.decoder(c_encoded)
        generated_imgs = self.deprocess_vgg19(generated_imgs)
        
        return generated_imgs
    
    @property
    def metrics(self):
        return [self.c_loss_metric]
    
    def deprocess_vgg19(self, imgs):
        # Ensure they are batched
        assert len(imgs.shape) == 4
        
        # Put back to 0...255
        imgs *= 255.0
        # Add mean
        imgs += [103.939, 116.779, 123.68]
        # BGR to RGB
        imgs = imgs[..., ::-1]
        # Clip
        imgs = tf.clip_by_value(imgs, 0.0, 255.0)
        # Cast
        imgs = tf.cast(imgs, tf.uint8)

        return imgs
    
    def save_architecture(self, log_dir):
        # Ensure the log_dir exists
        pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)

        with GFile(os.path.join(log_dir, 'style_transfer_architecture.json'), 'w') as f:
            f.write(self.to_json())
            
    def save_encoder(self, log_dir):
        self.encoder.save(log_dir)
    
    def save_model(self, log_dir, **kwargs):
        self.decoder.save(log_dir, **kwargs)

    @classmethod
    def load(cls, log_dir=None, epoch=None):
        model_found = bool(log_dir) and pathlib.Path(os.path.join(log_dir, 'style_transfer_architecture.json')).is_file()
                
        # If there isn't already a model create one from scratch and save it
        if not model_found:
            model = cls()
            if log_dir:
                model.save_architecture(log_dir)
                model.save_encoder(os.path.join(log_dir, 'encoder'))
            return model
        
        # Load the model's architecture
        with tf.keras.utils.custom_object_scope({'StyleTransfer':cls, 'Conv2DReflectivePadding':Conv2DReflectivePadding}):
            saved_json = GFile(os.path.join(log_dir, 'style_transfer_architecture.json'), 'r').read()
            model = tf.keras.models.model_from_json(saved_json)
            model.encoder = tf.keras.models.load_model(os.path.join(log_dir, 'encoder'))
        
        # If an epoch was provided, load the model at that epoch
        if epoch is not None:
            epoch_path = os.path.join(log_dir, 'weights', f'epoch_{epoch}')
            if not pathlib.Path(epoch_path).is_dir():
                print(f"Epoch {epoch} doesn't exists")
                return
            print('Loading Checkpoint:', epoch_path)
            model.decoder = tf.keras.models.load_model(epoch_path)
            model.decoder_compiled = True
        else:
            # Load the decoder's latest weights if there are any
            ckpts = glob.glob(os.path.join(log_dir, 'weights', '*'))
            if ckpts:
                latest_ckpt = max(ckpts, key=os.path.getmtime)
                print('Loading Checkpoint:', latest_ckpt)
                model.decoder = tf.keras.models.load_model(latest_ckpt)
                model.decoder_compiled = True

        return model
        
    def get_config(self):
        return {
            'encoder': self.encoder,
            'decoder': self.decoder
        }
    
    @classmethod
    def from_config(cls, config, **kwargs):
        encoder = tf.keras.models.model_from_json(json.dumps(config.pop('encoder')))
        decoder = tf.keras.models.model_from_json(json.dumps(config.pop('decoder')))

        style_transfer = cls(encoder=encoder, decoder=decoder)
        return style_transfer

#%%
def get_last_epoch(log_dir=None):
    if log_dir is None:
        return 0

    ckpts = glob.glob(os.path.join(log_dir, 'weights', '*'))

    # If there are no latest checkpoints start from scratch
    if not ckpts:
        return 0
    
    latest_ckpt_path = max(ckpts, key=os.path.getmtime)
    path = pathlib.PurePath(latest_ckpt_path)
    # Checkpoint prefixes are stored as epoch_x so we split to get the epoch number
    epoch = path.name.split('_')[-1]
    
    return int(epoch)

#%%
log_dir = None
# log_dir = 'testlogs/simple_adain/adam_255'

st = StyleTransfer.load(log_dir)
st.compile(
    optimizer=tf.keras.optimizers.Adam(lr=1e-02),
    content_loss=MeanSquaredError(reduction='none'),
)

# st.decoder.optimizer.lr = 1e-05

ds = get_dataset()

callbacks = [
#     EarlyStopping(monitor='loss', patience=3),
#     ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, cooldown=2),
#     TensorBoard(log_dir=f'{log_dir}/tensorboard', update_freq=100),
#     SaveModel(log_dir=f'{log_dir}/weights', update_freq=100),
#     SaveWeightsSummary(f'{log_dir}/tensorboard/weights')`,
#     LRDecay(1e-3, 5e-04, 1, 1000),
#     LRSearch(1e-3, 1e-07, 1, 300),
]

last_epoch = get_last_epoch(log_dir)

print('LR', st.decoder.optimizer.lr.numpy())

epochs = 300
st.fit(ds, epochs=epochs+last_epoch, initial_epoch=last_epoch, callbacks=callbacks)

#%%
st.decoder.optimizer.lr

#%%
st.decoder.optimizer.lr = 1e-05
last_epoch=40
epochs=50
st.fit(ds, epochs=epochs+last_epoch, initial_epoch=last_epoch, callbacks=callbacks)

#%%
class LRDecay(Callback):
    def __init__(self, lr_min, lr_max, steps, epochs, **kwargs):
        super().__init__(**kwargs)
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.steps = steps
        self.epochs = epochs
        self.decay = (lr_max / lr_min) ** (1/(epochs*steps))

    def on_train_begin(self, logs):
        self.model.decoder.optimizer.lr = self.lr = self.lr_min

    def on_train_batch_end(self, batch, logs):
        self.lr *= self.decay
        self.model.decoder.optimizer.lr = self.lr

#%%
class LRSearch(Callback):
    def __init__(self, lr_min, lr_max, steps, epochs, log_dir=None, save=False, **kwargs):
        super().__init__(**kwargs)
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.steps = steps
        self.epochs = epochs
        self.log_dir = log_dir
        self.save = save

        self.decay = (lr_max / lr_min) ** (1/(epochs*steps))
        self.history = {'lr': [], 'loss': []}
        self.best_loss = math.inf
        self.best_lr = math.inf

    def on_train_begin(self, logs):
        self.model.decoder.optimizer.lr = self.lr = self.lr_min

    def on_train_batch_end(self, batch, logs):
        self.lr *= self.decay
        self.model.decoder.optimizer.lr = self.lr
        
        self.history['lr'].append(self.lr)
        self.history['loss'].append(logs['c_loss'])
        
        if self.save:
            self.model.save_weights(os.path.join(self.log_dir, 'tmp_lr_search', str(len(self.history['loss']) - 1)))
        
        if self.best_loss > logs['c_loss']:
            self.best_loss = logs['c_loss']
            self.best_loss_index = len(self.history['loss']) - 1
            self.best_lr = self.history['lr'][self.best_loss_index - 1]

#%%
content_fn = glob.glob('/home/jephthia/datasets/mscoco/unlabeled2017/train/*')[0]

def parseim(file_name):
    img = tf.io.read_file(file_name)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224], antialias=True, method='nearest')
    img = tf.cast(img, tf.float32)
    img = tf.expand_dims(img, axis=0)

    return img

c_imgs = parseim(content_fn)

loadedst = StyleTransfer.load('testlogs/simple_adain/adam', epoch=3500)
pred = loadedst.predict(c_imgs, batch_size=1)

plt.imshow(tf.cast(c_imgs[0], tf.uint8))
plt.show()
plt.imshow(pred[0])
plt.show()

#%%
lr_min = 1e-03
lr_max = 1e-07
epochs = 100
steps = 1
decay = (lr_max / lr_min) ** (1/(epochs*steps))

lr = lr_min

his = []
for i in range(epochs):
    lr *= decay
    his.append(lr)
    
plt.plot(his)
plt.yscale('log')
