import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class Dataset():
    def __init__(self, train_path, test_path, input_width = 72, label_width = 216, shift = 216):
        df = pd.read_csv(train_path)
        self.nsamples = len(df)
        train_df = df.iloc[:int(self.nsamples * 0.75), 0:2]
        val_df = df.iloc[int(self.nsamples * 0.75):, 0:2]
        test_df = pd.read_csv(test_path).iloc[:, 0:2]

        self.num_features = train_df.shape[1]
        self.mean = train_df.mean()
        self.std = train_df.std()

        self.train_df = (train_df - self.mean) / self.std
        self.val_df = (val_df - self.mean) / self.std
        self.test_df = (test_df - self.mean) / self.std

        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.column_indices = {name: i for i, name in enumerate(self.train_df.columns)}

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
          data=data,
          targets=None,
          sequence_length=self.total_window_size,
          sequence_stride=1,
          shuffle=True,
          batch_size=32,)

        ds = ds.map(self.split_window)

        return ds

    def predict(self, inputs, labels, model=None, plot_col='Pipe 1'):
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]

        plt.ylabel(f'{plot_col}')
        plt.plot(self.input_indices, inputs[0, :, plot_col_index],
                 label='Inputs', marker='.', zorder=-10)

        label_col_index = plot_col_index
        predictions = model(inputs)
        plt.scatter(self.label_indices, labels[0, :, label_col_index], edgecolors='k', label='Labels', c='#2ca02c', s=64)
        plt.scatter(self.label_indices, predictions[0, :, label_col_index],marker='X', edgecolors='k', label='Predictions', c='#ff7f0e', s=64)
        plt.legend()

        plt.xlabel('time')
        plt.show()

        return predictions

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)
