#%%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Masking, Dense, Reshape, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.data import Dataset
from tensorflow.data.experimental import AUTOTUNE

# %%
class C:
    BATCH_SIZE = 10
    DS_LOG_DIR = '.'
    MASK_TOKEN = -1
    INIT_LR = 1e-03
    EPOCHS = 1

# %%
class DataManager:
    def __init__(self, log_dir):
        self.log_dir = log_dir

    def get_dataset(self):
        with np.load(os.path.join(self.log_dir, 'dataset.npz'), allow_pickle=True) as data:
            train_x = tf.ragged.constant(data['train_x'])
            train_y = data['train_y']

        ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        ds = ds.shuffle(100)
        ds = ds.batch(C.BATCH_SIZE, drop_remainder=True)
        ds = ds.map(lambda x, y: (x.to_tensor(C.MASK_TOKEN), y), num_parallel_calls=AUTOTUNE)
        return ds

# %%
class TrainManager:
    def __init__(self):
        self.dataset = None
        self.model = None

    def load_model(self, refresh=False):
        if self.model is None or refresh == True:
            self.model = SimpleBezier()
            self.model.compile(Adam(lr=C.INIT_LR), loss='mse')

        return self.model

    def load_dataset(self):
        if self.dataset is None:
            self.dataset = DataManager(log_dir=C.DS_LOG_DIR).get_dataset()

        return self.dataset

    def train(self, reload_model=False):
        self.load_dataset()
        self.load_model(refresh=reload_model)

        return self.model.fit(self.dataset, epochs=C.EPOCHS)

# %%
class SimpleBezier(Sequential):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add(Input((None, 2)))
        self.add(Masking(mask_value=C.MASK_TOKEN))
        self.add(GRU(128, return_sequences=True))
        self.add(GRU(128, return_sequences=True))
        self.add(GRU(128))
        self.add(Dense(128, activation='relu'))
        self.add(Dense(128, activation='relu'))
        self.add(Dense(64, activation='relu'))
        self.add(Dense(8, activation='sigmoid'))
        self.add(Reshape((4, 2)))


#%%
tm = TrainManager()

#%%
C.INIT_LR = 1e-03
C.EPOCHS = 50

hist = tm.train(reload_model=True)

# %%
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

# %%
x,y = next(iter(tm.dataset))
# x = x[:1]
# y = y[:1]

preds = tm.model.predict(x, batch_size=1)
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