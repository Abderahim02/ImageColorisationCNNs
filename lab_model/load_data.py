from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_io
from tensorflow_io.python.experimental.color_ops import rgb_to_lab



def preprocess_lab(image):

    image = tf.cast(image/255, tf.float32)
    lab = rgb_to_lab(image)
    l = lab[...,0]/100.0
    a = ( lab[...,1] + 128.0) /256.0
    b = ( lab[...,2] + 128.0) /256.0
    l = tf.expand_dims(l, axis=-1)
    a = tf.expand_dims(a, axis=-1)
    b = tf.expand_dims(b, axis=-1)
    ab = tf.concat([a,b], axis=-1)
    print(l.shape, ab.shape)
    return l, ab, l

def load_data(data_dir, image_size=(256, 256), batch_size=8):
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels=None,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True
    )
    colorization_dataset = dataset.map(preprocess_lab, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_size = int(0.2 * len(list(Path(data_dir).glob('*')))) // batch_size
    train_dataset = colorization_dataset.skip(val_size)
    val_dataset = colorization_dataset.take(val_size)
    train_dataset = train_dataset.shuffle(buffer_size=1000).prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset

def load_inference_data(data_dir, image_size=(256, 256), batch_size=8):
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels=None,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True
    )
    colorization_dataset = dataset.map(preprocess_lab, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_dataset = colorization_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return val_dataset