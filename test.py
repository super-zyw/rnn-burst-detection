import pandas as pd
import tensorflow as tf
import numpy as np
from PIL import Image
print(pd.__version__)
t = pd.read_csv("C:/Users/yinweizhang/Desktop/test.csv")
t = t.replace(r'\n','', regex=True)
t =  t.to_numpy().tolist()
print(t)
input_data_json = {"signature_name": "serving_default",
                   "instances": t}
print(input_data_json)
#df = pd.DataFrame({"A": [1, 2], "B": [3.0, 4.5]})
#print(df.to_numpy())
#im = tf.io.decode_image(tf.io.read_file('C:/Users/yinweizhang/Desktop/img.png'), channels=3) # make sure there's 3 colour channels (for PNG's)
#im = tf.cast(tf.expand_dims(im, axis=0), tf.int16)
#im = im.numpy().tolist()
#print(im)
