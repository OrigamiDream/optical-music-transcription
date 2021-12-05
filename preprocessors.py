import pandas as pd
import os
import cv2


def preprocess_features(ds_dir):
    df = pd.read_csv(os.path.join(ds_dir, 'notes.csv'))

    tick_to_wait = df['current_tick'] - df['current_tick'].shift(periods=1, fill_value=0)
    tick_to_wait = tick_to_wait.shift(periods=-1, fill_value=0)
    df.insert(1, 'tick_to_wait', tick_to_wait)

    df['tick_to_wait'] = df['tick_to_wait'] / 1000.
    df['offset_x_ratio'] = df['offset_x_ratio'].shift(periods=-1, fill_value=0)
    df['offset_y_ratio'] = df['offset_y_ratio'].shift(periods=-1, fill_value=0)
    df['transit_page'] = df['transit_page'].shift(periods=-1, fill_value=0)
    df['end_of_measure'] = df['end_of_measure'].shift(periods=-1, fill_value=False).astype('int')
    for i in range(128):
        col = 'changed-{}'.format(i)
        df[col] = df[col].astype('int')

        col = 'velocity-{}'.format(i)
        df[col] = df[col] / 127.
    df = df[:-1]  # drop last row
    return df


def preprocess_frame(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    img = 255 - img
    img = cv2.resize(img, (448, 448), interpolation=cv2.INTER_AREA)
    return img  # 448x448x1


def title_dirs_iterator(root_dir, dst_root_dir, verbose=True):
    for title_name in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, title_name)
        dst_dir_path = os.path.join(dst_root_dir, title_name)
        if not os.path.exists(dst_dir_path):
            os.makedirs(dst_dir_path)

        if os.path.isdir(dir_path):
            if verbose:
                print('Working on directory - {}'.format(title_name))
            yield dir_path, dst_dir_path


def preprocess_dataset_features(root_dir, dst_root_dir, verbose=True):
    for dir_path, dst_dir_path in title_dirs_iterator(root_dir, dst_root_dir, verbose=verbose):
        df = preprocess_features(dir_path)
        df.to_csv(os.path.join(dst_dir_path, 'note_features.csv'), index=False, sep=',')


def preprocess_dataset_images(root_dir, dst_root_dir, verbose=True):
    for dir_path, dst_dir_path in title_dirs_iterator(root_dir, dst_root_dir, verbose=verbose):
        index = 0
        for file_name in os.listdir(dir_path):
            if not file_name.endswith('.png'):
                continue
            img_path = os.path.join(dir_path, file_name)
            img = preprocess_frame(img_path)
            cv2.imwrite(os.path.join(dst_dir_path, file_name), img)
            index += 1
            if index % 100 == 0 and verbose:
                print('{} images have been preprocessed.'.format(index))

