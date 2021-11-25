import math
import numpy as np
from tensorflow.keras.utils import Sequence


class MultiOutputTimeseriesGenerator(Sequence):

    def __init__(self, data, targets, length, batch_size):
        self.data = data
        self.targets = targets
        self.length = length
        self.batch_size = batch_size

        for index, (name, target) in enumerate(targets.items()):
            if len(data) != len(target) + (length - 1):
                raise AssertionError(
                    '''
                    Data must be bigger than target {} by {}. Data length is {} while target {} length is {}
                    '''.format(name, length - 1, len(data), name, len(target)))

    def __len__(self):
        return math.ceil((len(self.data) - self.length + 1) / self.batch_size)

    def __getitem__(self, item):
        indices = np.arange(item * self.batch_size, min((item + 1) * self.batch_size, len(self.data) - self.length + 1))
        X, y = [], {}
        for name in self.targets.keys():
            y[name] = []
        for index in indices:  # create batch
            timeseries_X = []
            for i in np.arange(index, index + self.length):
                timeseries_X.append(self.data[i])

            for name, target in self.targets.items():
                y[name].append(target[index])
            X.append(np.array(timeseries_X))

        data_y = {}
        for key, value in y.items():
            data_y[key] = np.array(value, dtype=np.float32)
        X = np.array(X)
        return X, data_y


class ListMultiOutputTimeseriesGenerator(Sequence):

    def __init__(self, generators):
        self.generators = generators

        length = -1
        batch_size = -1
        for generator in self.generators:
            if length == -1 or batch_size == -1:
                length = generator.length
                batch_size = generator.batch_size
            else:
                assert length == generator.length and batch_size == generator.batch_size

    def __len__(self):
        length = 0
        for generator in self.generators:
            length += len(generator)
        return length

    def __getitem__(self, item):
        size = item
        for i, generator in enumerate(self.generators):
            length = len(generator)
            if item < length:
                return generator[item]
            else:
                item = length - item
        raise AssertionError('{} has been requested while the max length is {}'.format(size, self.__len__()))
