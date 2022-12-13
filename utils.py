import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
def compile_and_fit(model, window, MAX_EPOCHS:int=20, patience:int=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])
    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    model.save('saved_model/my_model')
    return history

@st.cache(suppress_st_warning=True)
def cusum(pre_arr, obs_arr, k: float = 0.2, thresh: float = 0.5) -> bool:
    """
    param pre_arr: 1*n, predicted values from RNN
    param obs_arr: 1*n, observed true values
    params
    """
    upper = [0]
    lower = [0]

    for num1, num2 in zip(pre_arr, obs_arr):
        resid = num1[0] - num2[0]
        upper.append(max(0, upper[-1] + resid - k))
        lower.append(min(0, lower[-1] + resid + k))

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(upper, c='#ff7f0e', label='C+')
    ax.plot(lower, c='#2ca02c', label='C-')

    ax.set_title('CUSUM control chart based on RNN prediction')
    ax.set_xlabel('time')
    ax.set_ylabel('residuals')
    ax.axhline(y = thresh, c = 'red', xmin = 0, xmax = len(upper), label='Upper Control Limit')
    ax.axhline(y = -thresh, c = 'red', xmin = 0, xmax = len(upper), label='Lower Control Limit')
    ax.legend()
    st.pyplot(fig)
    if max(upper) > thresh or min(lower) < -thresh:
        is_burst = True
    else:
        is_burst = False
    return is_burst

@st.cache(suppress_st_warning=True)
def ewma(pre_arr, obs_arr, alpha: float = 0.9, thresh: float = 0.5):
    """
    param pre_arr: 1*n, predicted values from RNN
    param obs_arr: 1*n, observed true values
    params
    """
    curr = [0]

    for num1, num2 in zip(pre_arr, obs_arr):
        resid = num1[0] - num2[0]
        curr.append(max(0, curr[-1] * (1-alpha) + resid * alpha))

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(curr, c='#ff7f0e', label='C+')

    ax.set_title('CUSUM control chart based on RNN prediction')
    ax.set_xlabel('time')
    ax.set_ylabel('residuals')
    ax.axhline(y = thresh, c = 'red', xmin = 0, xmax = len(curr), label='Upper Control Limit')
    ax.axhline(y = -thresh, c = 'red', xmin = 0, xmax = len(curr), label='Lower Control Limit')
    ax.legend()
    st.pyplot(fig)
    if max(curr) > thresh:
        is_burst = True
    else:
        is_burst = False
    return is_burst