import tensorflow as tf
import torch

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
if tf.config.experimental.list_physical_devices('GPU'):
    print("TensorFlow将自动使用GPU进行加速")
else:
    print("未发现可用的GPU，将使用CPU执行")



