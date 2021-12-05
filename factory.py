import numpy as np
import tensorflow.keras.utils as utils
import tensorflow.keras.backend as K

from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Bidirectional, GRU, Dense, TimeDistributed, Concatenate, Flatten
from tensorflow.keras.applications import VGG16


class OpticalMusicTranscriptionNetwork(Model):

    def __init__(self, timesteps=50, height=448, width=448, channels=1):
        super(OpticalMusicTranscriptionNetwork, self).__init__()

        self.compile_options = {
            'time': {
                'losses': 'mse',
                'metrics': ['mae']
            },
            'offset': {
                'losses': 'mse',
                'metrics': ['mae']
            },
            'condition': {
                'losses': 'binary_crossentropy',
                'metrics': ['accuracy']
            },
            'notes_changed': {
                'losses': 'binary_crossentropy',
                'metrics': ['accuracy']
            },
            'notes_velocity': {
                'losses': 'mse',
                'metrics': ['mae']
            }
        }

        print('Preparing network...')
        vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        vgg16.trainable = False

        input_layer = Input(shape=(timesteps, height, width, channels),
                            name='{}x{}x{}x{} captured pictures'.format(timesteps, height, width, channels))

        x = input_layer
        if channels == 1:
            x = Concatenate(axis=-1)([x, x, x])
        elif channels != 3:
            raise TypeError('channels must be 1 or 3.')

        for i in range(2):
            x = TimeDistributed(Conv2D(32,
                                       kernel_size=(3, 3),
                                       padding='same',
                                       kernel_initializer='he_normal',
                                       activation='relu',
                                       name='3x3_conv3d_32_{}'.format(i)))(x)
        x = TimeDistributed(MaxPooling2D((2, 2), strides=2, name='2x2_max_pooling2d'))(x)
        x = TimeDistributed(Conv2D(64, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block1_conv1'))(x)
        for i in range(2, len(vgg16.layers)):  # truncate input to block1_conv1
            layer = vgg16.layers[i]
            layer.trainable = False
            x = TimeDistributed(layer)(x)
        # x = TimeDistributed(vgg16)(x)

        # for i in range(2):
        #     x = TimeDistributed(Conv2D(512,
        #                                kernel_size=(3, 3),
        #                                padding='same',
        #                                kernel_initializer='he_normal',
        #                                activation='relu',
        #                                name='3x3_conv3d_32_{}'.format(i)))(x)
        # x = TimeDistributed(MaxPooling2D((2, 2), strides=2, name='2x2_max_pooling2d'))(x)
        x = TimeDistributed(Flatten())(x)
        x = self.gru_block(x, 256)

        time_out = self.time_layer(x)  # tick_to_wait
        offset_out = self.offset_layer(x)  # offset_x_ratio, offset_y_ratio
        condition_out = self.condition_layer(x)  # transit_page, end_of_measure
        notes_changed_out = self.notes_changed_layer(x)  # changed-0, changed-1, ..., changed-126, changed-127
        notes_velocity_out = self.notes_velocity_layer(x)  # velocity-0, velocity-1, ..., velocity-126, velocity-127
        self.model = Model(inputs=[input_layer],
                           outputs=[time_out, offset_out, condition_out, notes_changed_out, notes_velocity_out],
                           name='optical_music_transcription')

        print('Network prepared!')

    def load_model(self, filepath):
        print('Loading a model...')
        self.model = load_model(filepath)
        print('A model loaded!')

    def get_model(self) -> Model:
        return self.model

    def save_model_plot(self, filename):
        utils.plot_model(self.get_model(), to_file=filename, show_shapes=True)

    def get_params(self):
        model = self.get_model()
        num_trainable = np.sum([K.count_params(w) for w in model.trainable_weights])
        num_non_trainable = np.sum([K.count_params(w) for w in model.non_trainable_weights])
        return {
            'total_params': num_trainable + num_non_trainable,
            'trainable_params': num_trainable,
            'non_trainable_params': num_non_trainable
        }

    def choose_options(self, key) -> dict:
        options = {}
        for k, value in self.compile_options.items():
            options[k] = value[key]
        return options

    def choose_losses_options(self) -> dict:
        return self.choose_options('losses')

    def choose_metrics_options(self) -> dict:
        return self.choose_options('metrics')

    def compile_model(self):
        print('Compiling network...')
        self.get_model().compile(optimizer='adam',
                                 loss=self.choose_losses_options(),
                                 metrics=self.choose_metrics_options())

    def call(self, inputs, training=None, mask=None):
        return self.get_model()(inputs)

    def get_config(self):
        return self.get_model().get_config()

    @staticmethod
    def gru_block(x, mem_cells):
        x = Bidirectional(GRU(mem_cells, name='gru_{}_0'.format(mem_cells), return_sequences=True))(x)
        x = Bidirectional(GRU(mem_cells, name='gru_{}_1'.format(mem_cells), ))(x)
        return x

    @staticmethod
    def time_layer(x):
        x = Dense(32, name='dense_time_1', activation='relu')(x)
        x = Dense(1, name='time', activation='relu')(x)  # one output, regression
        return x

    @staticmethod
    def offset_layer(x):
        x = Dense(32, name='dense_offset_1', activation='relu')(x)
        x = Dense(2, name='offset')(x)
        return x

    @staticmethod
    def condition_layer(x):
        x = Dense(32, name='dense_condition_0', activation='relu')(x)
        x = Dense(2, name='condition', activation='sigmoid')(x)  # multiple output, multiple labels
        return x

    @staticmethod
    def notes_changed_layer(x):
        x = Dense(512, name='dense_notes_changed_0', activation='relu')(x)
        x = Dense(128, name='notes_changed', activation='sigmoid')(x)
        return x

    @staticmethod
    def notes_velocity_layer(x):
        x = Dense(512, name='dense_notes_velocity_0', activation='relu')(x)
        x = Dense(128, name='notes_velocity')(x)
        return x
