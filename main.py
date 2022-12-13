#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
from utils import compile_and_fit, cusum
from dataset import Dataset
import argparse

def train(train_path, test_path,IN_STEPS, OUT_STEPS, EPOCHS):

    DataLoader = Dataset(train_path, test_path, input_width=IN_STEPS, label_width=OUT_STEPS, shift=OUT_STEPS)
    train_df, val_df, test_df = DataLoader.train_df, DataLoader.val_df, DataLoader.test_df

    num_features = DataLoader.num_features
    multi_lstm_model = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, lstm_units].
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32, return_sequences=False),
        # Shape => [batch, out_steps*features].
        tf.keras.layers.Dense(OUT_STEPS*num_features,
                              kernel_initializer=tf.initializers.zeros()),
        # Shape => [batch, out_steps, features].
        tf.keras.layers.Reshape([OUT_STEPS, num_features])
    ])

    history = compile_and_fit(multi_lstm_model, DataLoader, MAX_EPOCHS = EPOCHS)

    test_window = tf.stack([np.array(test_df[:DataLoader.total_window_size])])
    test_inputs, test_labels = DataLoader.split_window(test_window)

    predicts = DataLoader.predict(test_inputs, test_labels, multi_lstm_model)

    cusum(predicts[0, :, 0].numpy(), test_labels[0, :, 0].numpy())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RNN-burst-detection')
    parser.add_argument('--train_path', type=str, default='F:/WaterLeakDetection/Code/matlab/RNN/train/normalpressure.csv')
    parser.add_argument('--test_path', type=str, default='F:/WaterLeakDetection/Code/matlab/RNN/test/burstpressure.csv')
    parser.add_argument('--IN_STEPS', type=int, default=72)
    parser.add_argument('--OUT_STEPS', type=int, default=216)
    parser.add_argument('--EPOCHS', type=int, default=20)

    args = parser.parse_args()
    train_path = args.train_path
    test_path = args.test_path
    IN_STEPS = args.IN_STEPS
    OUT_STEPS = args.OUT_STEPS
    EPOCHS = args.EPOCHS

    train(train_path, test_path, IN_STEPS, OUT_STEPS, EPOCHS)

#multi_lstm_model.evaluate(DataLoader.test, verbose=0)





