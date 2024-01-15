import os
import zipfile
import io
import numpy as np
import pandas as pd
import requests
import tensorflow as tf


def get_3d_flumy_data(filename="../data/3D/dataFlumy3D.csv", data_file='../data/3D',
                      dataset="../data/3D/dataFlumy3D.csv", max_simulation_nb=None):
    if not os.path.exists(data_file):
        os.mkdir(data_file)

    # Download data
    r = requests.get("https://soft.mines-paristech.fr/flumy-gan/data/dataFlumy3D.csv")
    content = r.content.decode('utf8')
    data = pd.read_csv(io.StringIO(content)).values

    if max_simulation_nb is not None:
        # To use if the whole dataset doesn't fit in RAM
        data = data[:, :max_simulation_nb]

    data = data.transpose(1, 0)

    data = data.reshape((-1, 20, 64, 128)) - 1
    # x_train = np.swapaxes(x_train.astype('int32'), 0, -1)
    return data


def load_data(nrows, ncols, filepath, sep=','):
    """
    Load vertical slices
    Parameters:
        nrows: number of rows per slice
        ncols: number of columns per slice
        filepath: path to the data file
        sep: .csv separator
    Returns:
        a one-hot encoded numpy array of size (samples_number, nrows, ncols, categories_number)
    """
    data = pd.read_csv(filepath, sep=sep).values

    # Number of categories and samples
    categories_number = int(data.max())
    samples_number = data.shape[1]

    data = data.transpose((1, 0))
    # reshaped_data = data.reshape((samples_number, nrows, ncols))
    one_hot = np.eye(categories_number, dtype='float32')[data - 1]

    return one_hot.reshape((samples_number, nrows, ncols, -1))


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size=32, dim=(64, 64, 4), shuffle=True):
        self.data = data
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.data)

    def __getitem__(self, index):
        X = self.data[index * self.batch_size:(index + 1) * self.batch_size]
        return X


class ConditionalDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size=32, dim=(64, 64, 4), wells=50, mode=0, shuffle=True):
        """

        modes: 0 = Random pixels wells (horizontal)
               1 = Grid pixels wells (horizontal, non-random) # ignores the parameter "wells"
               2 = Random pixels wells in only half the image (horizontal, variance test)
               3 = Random lines wells (vertical)
               4 = One central point # ignores the parameter "wells"
        Args:
            data:
            batch_size:
            dim:
            wells:
            mode:
            shuffle:
        """
        if mode not in [0, 1, 2, 3, 4]:
            raise ValueError("The mode must be 0, 1, 2, 3 or 4")

        self.data = data
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.wells = wells
        self.on_epoch_end()
        self.mode = mode

    def create_input_with_wells(self, data):
        X = np.zeros_like(data)
        mask = np.zeros((data.shape[0], data.shape[1], data.shape[2], 1))

        for i in range(self.batch_size):
            if self.mode == 1:
                x_coordinates = np.arange(4, self.dim[0], self.dim[0] // 4)
                y_coordinates = np.arange(8, self.dim[1], self.dim[1] // 8)
                n = len(x_coordinates)
                m = len(y_coordinates)
                x_coordinates = np.tile(x_coordinates, m)
                x_coordinates.sort()
                y_coordinates = np.tile(y_coordinates, n)

                X[i, x_coordinates, y_coordinates, :] = data[i, x_coordinates, y_coordinates, :]
                mask[i, x_coordinates, y_coordinates] = 1
            elif self.mode == 4:
                x_coordinates = np.asarray([self.dim[0] // 2, ])
                y_coordinates = np.asarray([self.dim[1] // 2, ])
                X[i, x_coordinates, y_coordinates, :] = data[i, x_coordinates, y_coordinates, :]
                mask[i, x_coordinates, y_coordinates] = 1
            else:
                if self.mode != 2:
                    random_y_coordinates = np.random.choice(self.dim[1], self.wells)
                else:
                    random_y_coordinates = np.random.choice(self.dim[1] // 2, self.wells)

                if self.mode != 3:
                    random_x_coordinates = np.random.choice(self.dim[0], self.wells)
                    X[i, random_x_coordinates, random_y_coordinates, :] = data[i, random_x_coordinates,
                                                                          random_y_coordinates, :]
                    mask[i, random_x_coordinates, random_y_coordinates] = 1

                else:
                    X[i, :, random_y_coordinates, :] = data[i, :, random_y_coordinates, :]
                    mask[i, :, random_y_coordinates] = 1

        return X, mask

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.data)

    def __getitem__(self, index):
        original = self.data[index * self.batch_size:(index + 1) * self.batch_size]
        X, mask = self.create_input_with_wells(original)
        return X, mask, original


class ConditionalDataGenerator3D(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size=32, dim=(20, 16, 64, 128, 4), wells=50, mode=0, shuffle=True):
        """

        modes: 0 = Random pixels wells (horizontal)
               1 = Grid pixels wells (horizontal, non-random)
               2 = Random pixels wells in only half the image (horizontal, variance test)
               3 = Random points (pixels)
        Args:
            data:
            batch_size:
            dim:
            wells:
            mode:
            shuffle:
        """
        if mode not in [0, 1, 2, 3]:
            raise ValueError("The mode must be 0, 1, 2 or 3")

        self.data = data
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.wells = wells
        self.on_epoch_end()
        self.mode = mode

    def create_input_with_wells(self, data):
        X = np.zeros_like(data)
        mask = np.zeros((data.shape[0], data.shape[1], data.shape[2], data.shape[3], 1))

        for i in range(self.batch_size):

            if self.mode == 1:
                x_coordinates = np.arange(4, self.dim[1], self.dim[1] // 2)
                y_coordinates = np.arange(8, self.dim[2], self.dim[2] // 4)
                n = len(x_coordinates)
                m = len(y_coordinates)
                x_coordinates = np.sort(np.tile(x_coordinates, m))
                y_coordinates = np.tile(y_coordinates, n)
                X[i, :, x_coordinates, y_coordinates, :] = data[i, :, x_coordinates, y_coordinates, :]
                mask[i, :, x_coordinates, y_coordinates] = 1
            else:
                if self.mode != 2:
                    random_y_coordinates = np.random.choice(self.dim[2], self.wells)
                else:
                    random_y_coordinates = np.random.choice(self.dim[2] // 2, self.wells)

                if self.mode != 3:
                    random_x_coordinates = np.random.choice(self.dim[1], self.wells)
                    X[i, :, random_x_coordinates, random_y_coordinates, :] = data[i, :, random_x_coordinates,
                                                                             random_y_coordinates, :]
                    mask[i, :, random_x_coordinates, random_y_coordinates] = 1

                else:
                    random_z_coordinates = np.random.choice(self.dim[0], self.wells)
                    random_x_coordinates = np.random.choice(self.dim[1], self.wells)
                    X[i, random_z_coordinates, random_x_coordinates, random_y_coordinates] = data[
                        i, random_z_coordinates, random_x_coordinates, random_y_coordinates]
                    mask[i, random_z_coordinates, random_x_coordinates, random_y_coordinates] = 1

        return X, mask

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.data)

    def __getitem__(self, index):
        original = self.data[index * self.batch_size:(index + 1) * self.batch_size]
        X, mask = self.create_input_with_wells(original)
        return X, mask, original
