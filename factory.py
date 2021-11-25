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
            'notes': {
                'losses': 'mse',
                'metrics': ['mae']
            }
        }

        print('Preparing network...')
        vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(448, 448, 3))
        vgg16.trainable = False

        input_layer = Input(shape=(timesteps, height, width, channels),
                            name='{}x{}x{}x{} captured pictures'.format(timesteps, height, width, channels))

        x = input_layer
        x = Concatenate(axis=-1)([x, x, x])
        x = TimeDistributed(vgg16)(x)
        for i in range(2):
            x = TimeDistributed(Conv2D(512,
                                       kernel_size=(3, 3),
                                       padding='same',
                                       kernel_initializer='he_normal',
                                       activation='relu',
                                       name='3x3_conv3d_32_{}'.format(i)))(x)
        x = TimeDistributed(MaxPooling2D((2, 2), strides=2, name='2x2_max_pooling2d'))(x)
        x = TimeDistributed(Flatten())(x)
        x = self.gru_block(x, 256)

        time_out = self.time_layer(x)  # tick_to_wait
        offset_out = self.offset_layer(x)  # offset_x_ratio, offset_y_ratio
        condition_out = self.condition_layer(x)  # transit_page, end_of_measure
        notes_out = self.notes_layer(x)  # changed-0, velocity-0, ..., changed-127, velocity-127
        self.model = Model(inputs=[input_layer],
                           outputs=[time_out, offset_out, condition_out, notes_out],
                           name='optical_music_transcription')

        print('Network prepared!')

    def load_model(self, filepath):
        print('Loading a model...')
        self.model = load_model(filepath)
        print('A model loaded!')

    def get_model(self) -> Model:
        return self.model

    def choose_losses_options(self) -> dict:
        options = {}
        for key, value in self.compile_options.items():
            options[key] = value['losses']
        return options

    def choose_metrics_options(self) -> dict:
        options = {}
        for key, value in self.compile_options.items():
            options[key] = value['metrics']
        return options

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
        x = Bidirectional(GRU(mem_cells, name='gru_{}_0'.format(mem_cells), dropout=0.2, activation='tanh', return_sequences=True))(x)
        x = Bidirectional(GRU(mem_cells, name='gru_{}_1'.format(mem_cells), dropout=0.2, activation='tanh', return_sequences=True))(x)
        x = Bidirectional(GRU(mem_cells, name='gru_{}_2'.format(mem_cells), dropout=0.2, activation='tanh', return_sequences=True))(x)
        x = Bidirectional(GRU(mem_cells, name='gru_{}_3'.format(mem_cells), dropout=0.2, activation='tanh'))(x)
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
    def notes_layer(x):
        x = Dense(512, name='dense_notes_0', activation='relu')(x)
        x = Dense(512, name='dense_notes_1', activation='relu')(x)
        x = Dense(256, name='notes')(x)
        return x

    @staticmethod
    def coordinate_loss(self, y_true, y_pred):
        pass
