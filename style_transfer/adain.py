#%%
import logging
logging.disable(logging.WARNING)

import os
import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt
import numpy as np
import glob
import json
import pathlib
import io
import h5py
import math

from tensorflow_io import IODataset
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
STYLE_WEIGHT = 0.8
# Layers to take from VGG19
STYLE_LAYERS = ('block1_conv1', 'block1_conv2', 'block1_pool', 'block2_conv1', 'block2_conv2', 'block2_pool', 'block3_conv1', 'block3_conv2',
                'block3_conv3', 'block3_conv4', 'block3_pool')
DECODER_LAYERS = ('block3_pool', 'block3_conv1', 'block2_pool', 'block2_conv1', 'block1_pool', 'block1_conv1')
# STYLE_LAYERS = ('block1_conv1', 'block1_conv2', 'block1_pool', 'block2_conv1', 'block2_conv2', 'block2_pool', 'block3_conv1', 'block3_conv2',
#                 'block3_conv3', 'block3_conv4', 'block3_pool', 'block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_conv4', 'block4_pool')

# Layers used to calculate the style loss
# LOSS_LAYERS = ('block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1')
LOSS_LAYERS = (['block1_conv1'])

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
    style_ds = build_ds(glob.glob('/home/jephthia/datasets/wikiart/train/*')[:1])
    val_content_ds = build_ds(glob.glob('/home/jephthia/datasets/mscoco/unlabeled2017/validate/*')[:2500])
    val_style_ds = build_ds(glob.glob('/home/jephthia/datasets/wikiart/validate/*')[:2500])
    
    # Train dataset
    ds = Dataset.zip((content_ds, style_ds))
    ds = ds.batch(BATCH_SIZE)
#     ds = ds.prefetch(2)
    
    # Validation dataset
    val_ds = Dataset.zip((val_content_ds, val_style_ds))
    val_ds = val_ds.batch(BATCH_SIZE)
#     val_ds = val_ds.cache()
#     val_ds = val_ds.prefetch(2)
   
    return ds, val_ds

#%%
vgg19 = VGG19()

# vgg19.summary()

#%%
class SaveModel(Callback):
    def __init__(self, log_dir, **kwargs):
        super(SaveModel, self).__init__(**kwargs)
        self.log_dir = log_dir
        
    def on_epoch_end(self, epoch, logs):
        self.model.save_model(log_dir=os.path.join(self.log_dir, f'epoch_{epoch+1}'))

#%%
class SaveWeightsSummary(Callback):
    def __init__(self, log_dir, **kwargs):
        super().__init__(**kwargs)
        self.log_dir = log_dir
        # Create directory if it doesn't exists
        pathlib.Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        
    def on_batch_end(self, batch, logs):
        for layer in self.model.decoder.layers:
            if layer.weights:
                with h5py.File(os.path.join(self.log_dir, 'weights.hdf5'), 'a') as f:
                    group = f[layer.name] if layer.name in f else f.create_group(layer.name, track_order=True)
                    kernel_group = group['kernel'] if 'kernel' in group else group.create_group('kernel')
                    bias_group = group['bias'] if 'bias' in group else group.create_group('bias')

                    kernel_group.create_dataset(str(batch), data=layer.weights[0].numpy(), dtype=np.float32)
                    bias_group.create_dataset(str(batch), data=layer.weights[1].numpy(), dtype=np.float32)                    

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
class StyleLoss(Loss):
    def __init__(self, *args, epsilon=1e-07, **kwargs):
        super().__init__(*args, **kwargs)
        self.mse = MeanSquaredError(reduction=self.reduction)
        self.epsilon = epsilon

    def call(self, g_encoded_outputs, s_encoded_outputs):
        mean_g, variance_g = tf.nn.moments(g_encoded_outputs, axes=[1, 2])
        mean_s, variance_s = tf.nn.moments(s_encoded_outputs, axes=[1, 2])

        std_g = tf.math.sqrt(variance_g + self.epsilon)
        std_s = tf.math.sqrt(variance_s + self.epsilon)

        loss = self.mse(mean_g, mean_s) + self.mse(std_g, std_s)

        return loss

#%%
class AdaIN(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, content, style, epsilon=1e-07):
        # Calculates the mean and variance of the content image
        c_mean, c_variance = tf.nn.moments(content, axes=[1, 2], keepdims=True)
        # Calculates the standard deviation of the content image
        c_std = tf.math.sqrt(c_variance + epsilon)
        
        # Calculates the mean and variance of the style image
        s_mean, s_variance = tf.nn.moments(style, axes=[1, 2], keepdims=True)
        # Calculates the standard deviation of the style image
        s_std = tf.math.sqrt(s_variance + epsilon)

        norm = (s_std * ((content - c_mean) / c_std)) + s_mean

        return norm

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
        self.s_loss_metric = Mean(name='s_loss')
        self.loss_metric = Mean(name='loss')

    def compile(self, optimizer, content_loss, style_loss, **kwargs):
        super().compile(**kwargs)
        if not getattr(self, 'decoder_compiled', False):
             self.decoder.compile(optimizer=optimizer)
        self.content_loss = content_loss
        self.style_loss = style_loss
        self.adain = AdaIN()
    
    @tf.function
    def train_step(self, data, training=True):
        c_encoded_outputs, s_encoded_outputs = data

        # The outputs of the encoded style images and the encoded generated images
        # Retrieved from the selected encoder layers used for the loss
        s_loss_outputs = []
        g_loss_outputs = []
            
        with tf.GradientTape(watch_accessed_variables=training) as tape:
            # 1. Encode the content and style image
            for layer in self.encoder.layers:
                # Encode the content image
                c_encoded_outputs = layer(c_encoded_outputs)
                # Encode the style image
                s_encoded_outputs = layer(s_encoded_outputs)

                # If this layer is used to calculate the loss save its outputs
                if layer.name in LOSS_LAYERS:
                    s_loss_outputs.append(s_encoded_outputs)

            # 2. Adaptive Instance Normalization
            adain_outputs = self.adain(c_encoded_outputs, s_encoded_outputs)

            # 3. Decode the feature maps generated by AdaIN to get the final generated image
            generated_imgs = self.decoder(adain_outputs, training=training)

            # 4. Encode the generated image to calculate the loss
            g_encoded_outputs = generated_imgs
            for layer in self.encoder.layers:
                # Encode the generated image
                g_encoded_outputs = layer(g_encoded_outputs)

                # If this layer is used to calculate the loss save its outputs
                if layer.name in LOSS_LAYERS:
                    g_loss_outputs.append(g_encoded_outputs)
                    
            # 5. Calculate the content loss
            c_per_replica_loss = self.content_loss(g_encoded_outputs, adain_outputs) # (N,W,H)
            # Reduce the loss (we do this ourselves in order to be compatible with distributed training)
            global_c_loss_size = tf.size(c_per_replica_loss) * self.distribute_strategy.num_replicas_in_sync
            global_c_loss_size = tf.cast(global_c_loss_size, dtype=tf.float32)
            c_loss = tf.nn.compute_average_loss(c_per_replica_loss, global_batch_size=global_c_loss_size)

            assert len(g_loss_outputs) == len(s_loss_outputs)

            # 6. Calculate style loss
            s_loss = 0
            for i in range(len(g_loss_outputs)):
                s_per_replica_loss = self.style_loss(g_loss_outputs[i], s_loss_outputs[i]) # (N,)
                # Reduce the loss (we do this ourselves in order to be compatible with distributed training)
                global_s_loss_size = BATCH_SIZE * self.distribute_strategy.num_replicas_in_sync
                s_loss += tf.nn.compute_average_loss(s_per_replica_loss, global_batch_size=global_s_loss_size)

            # 7. Calculate the loss
            loss = c_loss + s_loss*STYLE_WEIGHT

        # 8. Apply gradient descent
        if training:
            gradients = tape.gradient(loss, self.decoder.trainable_variables)
                    
#             gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
#             tf.print('---')
#             tf.print('glonorm', tf.linalg.global_norm(gradients))
#             tf.print(list((i, tf.math.reduce_min(n), tf.math.reduce_max(n)) for i,n in enumerate(gradients)))
#             tf.print(list((i, tf.math.reduce_min(n), tf.math.reduce_max(n)) for i,n in enumerate(self.decoder.trainable_variables)))
#             tf.print('C_ENC --> ', tf.math.reduce_min(c_encoded_outputs), tf.math.reduce_max(c_encoded_outputs))
#             tf.print('S_ENC --> ', tf.math.reduce_min(s_encoded_outputs), tf.math.reduce_max(s_encoded_outputs))
#             tf.print('ADAIN --> ', tf.math.reduce_min(adain_outputs), tf.math.reduce_max(adain_outputs))
#             tf.print('GEN_IMG --> ', tf.math.reduce_min(generated_imgs), tf.math.reduce_max(generated_imgs))
#             tf.print('G_ENC --> ', tf.math.reduce_min(g_encoded_outputs), tf.math.reduce_max(g_encoded_outputs))
#             tf.print('S_LOSS_E --> ', tf.math.reduce_min(s_loss_outputs[i]), tf.math.reduce_max(s_loss_outputs[i]))
#             tf.print('G_LOSS_E --> ', tf.math.reduce_min(g_loss_outputs[i]), tf.math.reduce_max(g_loss_outputs[i]))

            self.decoder.optimizer.apply_gradients(zip(gradients, self.decoder.trainable_variables))
        
        # 9. Update the metrics
        self.c_loss_metric.update_state(c_loss)
        self.s_loss_metric.update_state(s_loss)
        self.loss_metric.update_state(loss)

        return { m.name: m.result() for m in self.metrics }

    @tf.function
    def test_step(self, data):
        c_encoded_outputs, s_encoded_outputs = data
        
        # The outputs of the encoded style images and the encoded generated images
        # Retrieved from the selected encoder layers used for the loss
        s_loss_outputs = []
        g_loss_outputs = []
            
        # 1. Encode the content and style image
        for layer in self.encoder.layers:
            # Encode the content image
            c_encoded_outputs = layer(c_encoded_outputs, training=False)
            # Encode the style image
            s_encoded_outputs = layer(s_encoded_outputs, training=False)

            # If this layer is used to calculate the loss save its outputs
            if layer.name in LOSS_LAYERS:
                s_loss_outputs.append(s_encoded_outputs)

        # 2. Adaptive Instance Normalization
        adain_outputs = self.adain(c_encoded_outputs, s_encoded_outputs)

        # 3. Decode the feature maps generated by AdaIN to get the final generated image
        generated_imgs = self.decoder(adain_outputs, training=False)

        # 4. Encode the generated image to calculate the loss
        g_encoded_outputs = generated_imgs
        for layer in self.encoder.layers:
            # Encode the generated image
            g_encoded_outputs = layer(g_encoded_outputs, training=False)

            # If this layer is used to calculate the loss save its outputs
            if layer.name in LOSS_LAYERS:
                g_loss_outputs.append(g_encoded_outputs)

        # 5. Calculate the content loss
        c_per_replica_loss = self.content_loss(g_encoded_outputs, adain_outputs) # (N,W,H)
        # Reduce the loss (we do this ourselves in order to be compatible with distributed training)
        global_c_loss_size = tf.size(c_per_replica_loss) * self.distribute_strategy.num_replicas_in_sync
        global_c_loss_size = tf.cast(global_c_loss_size, dtype=tf.float32)
        c_loss = tf.nn.compute_average_loss(c_per_replica_loss, global_batch_size=global_c_loss_size)

        assert len(g_loss_outputs) == len(s_loss_outputs)

        # 6. Calculate style loss
        s_loss = 0
        for i in range(len(g_loss_outputs)):
            s_per_replica_loss = self.style_loss(g_loss_outputs[i], s_loss_outputs[i]) # (N,)
            # Reduce the loss (we do this ourselves in order to be compatible with distributed training)
            global_s_loss_size = BATCH_SIZE * self.distribute_strategy.num_replicas_in_sync
            s_loss += tf.nn.compute_average_loss(s_per_replica_loss, global_batch_size=global_s_loss_size)

        # 7. Calculate the loss
        loss = c_loss + s_loss*STYLE_WEIGHT

        # 9. Update the metrics
        self.c_loss_metric.update_state(c_loss)
        self.s_loss_metric.update_state(s_loss)
        self.loss_metric.update_state(loss)
        
        return { m.name: m.result() for m in self.metrics }
        
    @tf.function
    def predict_step(self, data):
        content_imgs, style_imgs = data[0]
        
        # Ensure these are batched
        assert len(content_imgs.shape) == 4
        assert len(style_imgs.shape) == 4

        content_imgs = vgg19_preprocess_input(content_imgs) / 255.0
        style_imgs = vgg19_preprocess_input(style_imgs) / 255.0

        c_encoded = content_imgs
        s_encoded = style_imgs

        # Encode the contents and styles
        for layer in self.encoder.layers:
            c_encoded = layer(c_encoded)
            s_encoded = layer(s_encoded)
            
        # Apply adaptive batch normalization
        adain_outputs = AdaIN()(c_encoded, s_encoded)
        
        # Decode the images to generate them
        generated_imgs = self.decoder(adain_outputs)
        generated_imgs = self.deprocess_vgg19(generated_imgs)
        
        return generated_imgs
    
    @property
    def metrics(self):
        return [self.loss_metric, self.c_loss_metric, self.s_loss_metric]
    
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
    def load(cls, log_dir=None):
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
# log_dir = None
log_dir = 'testlogs/one_value_corrected_c_loss'

st = StyleTransfer.load(log_dir)
st.compile(
    optimizer=tf.keras.optimizers.SGD(lr=1e-07),
    content_loss=MeanSquaredError(reduction='none'),
    style_loss=StyleLoss(reduction='none')
)

# st.decoder.optimizer.lr = 0.0
st.decoder.optimizer.lr = 4e-07

# st.decoder = tf.keras.models.load_model(os.path.join(log_dir, 'tmp_lr_search', str(lrsearch.best_loss_index)))
# st.decoder.optimizer.lr = lrsearch.history['lr'][lrsearch.best_loss_index - 1]

ds, val_ds = get_dataset()

callbacks = [
#     EarlyStopping(monitor='loss', patience=3),
#     ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, cooldown=2),
#     TensorBoard(log_dir=f'{log_dir}/tensorboard'),
    SaveModel(log_dir=f'{log_dir}/weights')
#     SaveWeightsSummary(f'{log_dir}/tensorboard/weights')`
#     LRSearch(1e-10, 1e-01, 1, 100, log_dir, save=False)
]

last_epoch = get_last_epoch(log_dir)

print('LR', st.decoder.optimizer.lr.numpy())

epochs = 2000
st.fit(ds, epochs=epochs+last_epoch, initial_epoch=last_epoch, callbacks=callbacks)

#%%
class LRSearch(Callback):
    def __init__(self, lr_min, lr_max, steps, epochs, log_dir, save=False, **kwargs):
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
        self.history['loss'].append(logs['loss'])
        
        if self.save:
            self.model.save_weights(os.path.join(self.log_dir, 'tmp_lr_search', str(len(self.history['loss']) - 1)))
        
        if self.best_loss > logs['loss']:
            self.best_loss = logs['loss']
            self.best_loss_index = len(self.history['loss']) - 1
            self.best_lr = self.history['lr'][self.best_loss_index - 1]

#%%
lr_min = 1e-06
lr_max = 1e-02
steps = 1
epochs = 50

decay = (lr_max / lr_min) ** (1/(epochs*steps))


lr = lr_min

x = list(i+1 for i in range(epochs))
y = []

for i in x:
    lr *= decay
    y.append(lr)

plt.plot(x,y,'-')
plt.ylabel('lr')
plt.xlabel('steps')
plt.show()

#%%
np.max(lrsearch.history['loss'])

#%%
lrsearch.history['lr']

#%%
# ran = list(i for i in lrsearch.history['loss'] if i > 140 and i < 150)
# len(ran)

min(lrsearch.history['loss'])
lrsearch.history['loss'].index(97.63299560546875)
lrsearch.history['lr'][47]

#%%
# prevsearch = []

#%%
# prevsearch.append(callbacks[0])

#%%
np.max(lrsearch.history['loss'])
lrsearch.history['loss'].index(3296092657549312.0)

#%%
lrsearch = prevsearch[-1]

fig = plt.figure(figsize=(15,8))
start = 0
end = 124
plt.plot(lrsearch.history['lr'][start:end], lrsearch.history['loss'][start:end], '-')
plt.ylabel('loss')
plt.xlabel('lr')
plt.show()

#%%
best_descent = math.inf
best_lr = math.inf
best_loss = math.inf
best_loss_index = None

# Find best loss
for i in range(len(lrsearch.history['loss'])):
    if best_loss > lrsearch.history['loss'][i]:
        best_loss = lrsearch.history['loss'][i]
        best_loss_index = i

# for i in range(1, len(lrsearch.history['lr'])):
#     slope = lrsearch.history['loss'][i] - lrsearch.history['loss'][i-1]
#     if best_descent > slope:
#         best_descent = slope
#         best_lr = lrsearch.history['lr'][i - 1]
#         loss = lrsearch.history['loss'][i]
#         index = i
        
# print('best_descent', best_descent)
# print('best_lr', best_lr)
print('best_loss', best_loss)
print('best_loss_index', best_loss_index)

#%%
print('best_lr', lrsearch.best_lr)
print('best_loss', lrsearch.best_loss)
print('best_loss_index', lrsearch.best_loss_index)

#%%
start = 20
end = 30
plt.plot(lrsearch.history['lr'][start:end], lrsearch.history['loss'][start:end], 'o-')
plt.ylabel('loss')
plt.xlabel('lr')
plt.show()

#%%
lrsearch.history['loss'][27]

#%%
for i,l in enumerate(st.decoder.layers):
    print(i,l.name)
    if 'conv' in l.name:
        plt.hist(l.weights[0].numpy().flatten())
        plt.show()

#%%
for i,l in enumerate(vgg19.layers):
    print(i,l.name)
    if 'conv' in l.name:
        plt.hist(l.weights[0].numpy().flatten())
        plt.show()

#%%
content_fn = glob.glob('/home/jephthia/datasets/mscoco/unlabeled2017/train/*')[0]
style_fn = glob.glob('/home/jephthia/datasets/wikiart/train/*')[0]
print(content_fn)
print(style_fn)

def parseim(file_name):
    img = tf.io.read_file(file_name)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224], antialias=True, method='nearest')
    img = tf.cast(img, tf.float32)
    img = tf.expand_dims(img, axis=0)

    return img

c_imgs = parseim(content_fn)
s_imgs = parseim(style_fn)

# print(c_imgs)

pred = st.predict((c_imgs, s_imgs), batch_size=1)
im = pred[0]

print('min', np.min(im))
print('max', np.max(im))
print('avg', np.mean(im))

# print(im)
# plt.hist(im.flatten())
# plt.show()
plt.hist(im.flatten())
plt.show()
c_imgs = tf.cast(c_imgs, tf.uint8)
s_imgs = tf.cast(s_imgs, tf.uint8)

plt.imshow(c_imgs[0])
plt.show()
plt.imshow(s_imgs[0])
plt.show()
plt.imshow(im)
plt.show()

#%%
mycimg, mysimg = next(iter(ds))

print('c image', np.min(mycimg.numpy()), np.max(mycimg.numpy()))
print('s image', np.min(mysimg.numpy()), np.max(mysimg.numpy()))

plt.hist(mycimg.numpy().flatten())
plt.show()

for l in st.encoder.layers:
    mycimg = l(mycimg)
    mysimg = l(mysimg)
    
print('encoded c image', np.min(mycimg.numpy()), np.max(mycimg.numpy()))
print('encoded s image', np.min(mysimg.numpy()), np.max(mysimg.numpy()))

myada = AdaIN()(mycimg, mysimg)

print('adain', np.min(myada.numpy()), np.max(myada.numpy()))

# declayers = iter(st.decoder.layers)

gen = st.decoder(myada)

# gen = next(declayers)(myada) # up
# gen = next(declayers)(gen) # conv
# gen = next(declayers)(gen) # up
# gen = next(declayers)(gen) # conv
# gen = next(declayers)(gen) # up
# gen = next(declayers)(gen) # conv
# gen = next(declayers)(gen) # output

# for u in st.decoder.layers:
#     if 'conv' in u.name:
#         plt.hist(u.weights[0].numpy().flatten())
#         plt.show()

plt.hist(gen.numpy().flatten())
plt.show()
print('generated', np.min(gen.numpy()), np.max(gen.numpy()))


finalimg = st.deprocess_vgg19(gen)
# finalimg = gen * 255.0
# finalimg += [103.939, 116.779, 123.68]
# finalimg = finalimg[..., ::-1]
# finalimg = tf.clip_by_value(finalimg, 0.0, 255.0)
# finalimg = tf.cast(finalimg, tf.uint8)

plt.hist(finalimg.numpy().flatten())
plt.show()

plt.imshow(finalimg[0])
plt.show()
