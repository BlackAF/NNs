#%%
import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Embedding, Dense
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.io.gfile import GFile
from tensorflow.strings import unicode_split
from tensorflow.data import Dataset

tf.config.list_physical_devices('GPU')

#%%
class DataManager():    
    def __init__(self, file_path):
        self.file_path = file_path
        self.ids_to_chars_layer = None
    
    def decode_text(self, ids):
        if self.ids_to_chars_layer is None:
            return None

        return tf.strings.reduce_join(self.ids_to_chars_layer(ids), axis=-1)

    def load_data(self):
        data = GFile(self.file_path, 'rb').read().decode(encoding='UTF-8')

        # Get a list of the unique characters in the text
        vocab = list(sorted(set(data)))
        vocab_size = len(vocab)

        chars_to_ids = StringLookup(vocabulary=vocab)
        self.ids_to_chars_layer = StringLookup(vocabulary=chars_to_ids.get_vocabulary(), invert=True)

        # Split the entire text by character
        chars = unicode_split(data, 'UTF-8')
        ids_of_chars = chars_to_ids(chars)

        # Group characters to form sequences (+1 since the targets are shifted by one)
        sequences_ds = Dataset.from_tensor_slices(ids_of_chars)
        sequences_ds = sequences_ds.batch(C.SEQUENCE_LENGTH+1)

        # Batch the sequences
        ds = sequences_ds.padded_batch(C.BATCH_SIZE)
        ds = ds.map(self._to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.shuffle(C.BUFFER_SIZE)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        return ds
    
    def _to_inputs_and_targets(self, sequences):
        # Exclude the last character
        inputs = sequences[:, :-1] # H e l l o -> H e l l
        # Exclude the first character
        targets = sequences[:, 1:] # H e l l o -> e l l o
        return inputs, targets

#%%
class TrainManager():
    def __init__(self):
        self.model = None
        self.dataset = None
        self.data_manager = DataManager(C.DATA_PATH)
        
    def load_data(self):
        self.dataset = self.data_manager.load_data()
    
    def load_model(self):
        self.model = Sequential()
        self.model.add(Embedding(input_dim=C.VOCAB_SIZE, output_dim=C.EMBEDDING_DIM, input_length=C.SEQUENCE_LENGTH))
        self.model.add(SimpleRNN(128, return_sequences=True))
        self.model.add(Dense(C.VOCAB_SIZE))

        self.model.compile(
            optimizer=Adam(lr=C.INITIAL_LR),
            loss=SparseCategoricalCrossentropy(from_logits=True)
        )
        
    def sample_text(self):
        preds = self.model.predict(self.dataset)

        print('preds shape', preds.shape)

        sampled_indices = tf.random.categorical(preds[0], num_samples=1)
        sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
        pred_text = self.data_manager.decode_text(sampled_indices)

        print('pred', pred_text.numpy())
        
    def train(self):
        if self.model is None:
            self.load_model()
            
        if self.dataset is None:
            self.load_data()
            
        self.model.fit(self.dataset, epochs=C.EPOCHS)

#%%
class C:
    # --- DATA --- #
    SEQUENCE_LENGTH = 100
    BATCH_SIZE = 8
    VOCAB_SIZE = 67
    EMBEDDING_DIM = 56
    BUFFER_SIZE = 10000
    DATA_PATH = 'datasets/simpleline.svg'
    
    # -- TRAINING -- #
    EPOCHS = 100
    INITIAL_LR = 1e-02

tm = TrainManager()
tm.train()
