from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import transform


def preprocess_for_colorization(image):
    grayscale = tf.image.rgb_to_grayscale(image) 

    return grayscale/255.0, image/255.0

def preprocess_with_mask(image, num_hints):
    grayscale = tf.image.rgb_to_grayscale(image)  
    print(image.shape)
    height, width = image.shape[1], image.shape[2]

    # Create a binary mask with num_hints pixels randomly set to 1
    num_pixels = height * width
    flat_indices = tf.random.shuffle(tf.range(num_pixels))[:num_hints]
    binary_mask = tf.scatter_nd(
        indices=tf.expand_dims(flat_indices, axis=1),
        updates=tf.ones_like(flat_indices, dtype=tf.float32),
        shape=(num_pixels,)
    )
    binary_mask = tf.reshape(binary_mask, (height, width))  # Reshape to 2D

    # Expand binary mask to 3 channels and apply it to the original image
    binary_mask = tf.expand_dims(binary_mask, axis=-1)
    color_mask = image * tf.cast(binary_mask, dtype=image.dtype)

    grayscale = grayscale / 255.0
    color_mask = color_mask / 255.0
    image = image / 255.0

    return (grayscale, color_mask), image


def load_data(data_dir, model_type, image_size=(256, 256), batch_size=8, num_hints=10):
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels=None,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True
    )
    if model_type == "mask":
        num_pixels = image_size[0] * image_size[1]
        num_hints = num_hints
        colorization_dataset = dataset.map(lambda x: preprocess_with_mask(x, num_hints), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif model_type == "mse" or model_type == "perc":
        colorization_dataset = dataset.map(preprocess_for_colorization, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    val_size = int(0.2 * len(list(Path(data_dir).glob('*')))) // batch_size
    train_dataset = colorization_dataset.skip(val_size)
    val_dataset = colorization_dataset.take(val_size)
    train_dataset = train_dataset.shuffle(buffer_size=1000).prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset


def load_inference_data(data_dir, model_type, num_hints = 10, image_size=(256, 256), batch_size=8):
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels=None,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True
    )

    if model_type == "mask":
        colorization_dataset = dataset.map(lambda x: preprocess_with_mask(x, num_hints), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else :
        colorization_dataset = dataset.map(preprocess_for_colorization, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_dataset = colorization_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return val_dataset