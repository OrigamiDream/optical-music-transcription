import os
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sequence import MultiOutputTimeseriesGenerator, ListMultiOutputTimeseriesGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from factory import OpticalMusicTranscriptionNetwork

print()

root_dir = 'bin_datasets'
train1_ds_dir = os.path.join(root_dir, 'Sheets0')
train2_ds_dir = os.path.join(root_dir, 'Sheets1')
valid_ds_dir = os.path.join(root_dir, 'Sheets2')

timesteps = 10
batch_size = 4


def load_data(dir_path, load_only=None) -> (np.ndarray, pd.DataFrame):
    X, y = None, None
    if load_only is None or load_only.upper() == 'X':
        img_paths = glob.glob(os.path.join(dir_path, '*.png'))  # time-series
        img_paths.sort()
        X = [np.asarray(cv2.imread(path, cv2.IMREAD_GRAYSCALE), dtype=np.uint8) for path in img_paths]
        for i in range(timesteps - 1):
            empty_image = np.zeros(X[0].shape)
            X.insert(0, empty_image)
        X = np.array(X)
        X = X / 255.
    if load_only is None or load_only.upper() == 'Y':
        y = pd.read_csv(os.path.join(dir_path, 'note_features.csv'))
    return X, y


def create_timeseries_iterator(dir_path) -> (np.ndarray, MultiOutputTimeseriesGenerator):
    data_X, data_y = load_data(dir_path)

    time_data_y = data_y[['tick_to_wait']]
    offset_data_y = data_y[['offset_x_ratio', 'offset_y_ratio']]
    condition_data_y = data_y[['transit_page', 'end_of_measure']]
    notes_changed = []
    notes_velocity = []
    for i in range(128):
        notes_changed.append('changed-{}'.format(i))
        notes_velocity.append('velocity-{}'.format(i))
    notes_changed_y = data_y[notes_changed]
    notes_velocity_y = data_y[notes_velocity]

    targets = {
        'time': time_data_y.to_numpy(),
        'offset': offset_data_y.to_numpy(),
        'condition': condition_data_y.to_numpy(),
        'notes_changed': notes_changed_y.to_numpy(),
        'notes_velocity': notes_velocity_y.to_numpy()
    }

    timeseries_iter = MultiOutputTimeseriesGenerator(data_X,
                                                     targets,
                                                     length=timesteps,
                                                     batch_size=batch_size)
    return data_X, timeseries_iter


print('Creating timeseries iterators...')
_, train_iter1 = create_timeseries_iterator(train1_ds_dir)
_, train_iter2 = create_timeseries_iterator(train2_ds_dir)
train_iter = ListMultiOutputTimeseriesGenerator([train_iter1, train_iter2])
valid_X, valid_iter = create_timeseries_iterator(valid_ds_dir)
print('Timeseries iterators have been created.')

filepath = 'best_weights.hdf5'
callbacks = [EarlyStopping(monitor='val_loss', patience=50, verbose=1),
             ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, verbose=1)]

omr = OpticalMusicTranscriptionNetwork(timesteps=timesteps)
# omr.save_model_plot(filename='omt-model.png')
if os.path.exists(filepath):
    omr.load_model(filepath)
else:
    omr.compile_model()

probe = omr.get_model().fit(train_iter,
                            validation_data=valid_iter,
                            validation_freq=1,
                            steps_per_epoch=len(train_iter),
                            epochs=5000,
                            callbacks=callbacks,
                            use_multiprocessing=True)

fig = plt.figure(figsize=(14, 7))
keys = probe.history.keys()
for i, key in enumerate(keys):
    ax = fig.add_subplot(len(keys), 1, i + 1)
    ax.plot(probe.history[key], label=key)
    ax.set_xlabel('Epochs')
    ax.set_ylabel(key)
plt.tight_layout()
plt.legend()
plt.show()


def choose_images_to_eval(dir_path, from_tick, to_tick):
    df = pd.read_csv(os.path.join(dir_path, 'note_features.csv'))
    mask = (df['current_tick'] >= from_tick) & (df['current_tick'] <= to_tick)
    test_df = df[mask]
    num_features = len(test_df)

    ticks = test_df['current_tick']

    first_index = test_df.index[0] - (timesteps - 1)
    last_index = test_df.index[-1]

    test_df = df.iloc[first_index:last_index + 1, 0]
    img_paths = [os.path.join(dir_path, 'frame-%08d.png' % current_tick) for current_tick in test_df.values]
    img_paths.sort()
    X = [np.asarray(cv2.imread(path, cv2.IMREAD_GRAYSCALE), dtype=np.uint8) for path in img_paths]
    X = np.array(X)
    X = X / 255.
    return X, num_features, ticks.values


def create_timeseries_tensor(data, num_indices):
    X = []
    for index in range(num_indices):
        timeseries_X = []
        for j in np.arange(index, index + timesteps):
            timeseries_X.append(data[j])
        X.append(timeseries_X)
    X = np.array(X)
    return X


print('Choosing images')
test_X, features, ticks = choose_images_to_eval(dir_path=train1_ds_dir, from_tick=32159, to_tick=37780)
test_X = create_timeseries_tensor(test_X, features)

test_one = test_X[100]
test_one_step = test_one[0] * 255

plt.imshow(test_one_step, cmap=plt.cm.gray)
plt.show()

from_index = 0
steps = 50

print('Predicting a simple timeseries tensor')
result = omr.get_model().predict(test_X[from_index:from_index + steps])
with open('res.csv', 'w') as csv_file:
    csv_file.write('current_tick,tick_to_wait,offset_x_ratio,offset_y_ratio,transit_page,end_of_measure\n')
    for i in range(steps):
        time = result[0][i]
        offset = result[1][i]
        condition = result[2][i]
        csv_file.write(f'{ticks[i]},{time[0]},{offset[0]},{offset[1]},{condition[0]},{condition[1]}\n')
