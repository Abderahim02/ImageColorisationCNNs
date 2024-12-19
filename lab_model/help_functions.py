import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt
from keras.callbacks import Callback
import json
import tensorflow_io
from PIL import Image

import datetime


def psnr(y_true, y_pred):
    max_pixel = 1.0  
    # print(y_true.shape, y_pred.shape)
    return tf.image.psnr(y_true, y_pred, max_val=max_pixel)

def ssim(y_true, y_pred):
    max_pixel = 1.0  
    return tf.image.ssim(y_true, y_pred, max_val=max_pixel)
def save_loss(history, path):
    with open(path + ".json", 'w') as f:
        json.dump(history.history, f)
        print(f"Model metrics savec at {path + ".json"} ")

def train_model(model, train_dataset, val_dataset, save_path, save_loss_path, n_epochs = 5, save_interval = 1):
    save_callback = SaveModelEveryNEpochs(save_path=save_path, interval=save_interval)

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=n_epochs,
        callbacks=[save_callback]
    )
    save_loss(history, save_loss_path)


# Charger une image grayscale pour prédiction
def load_grayscale_image(image_path, target_size):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=1)  
    image = tf.image.resize(image, target_size) / 255.0  
    return tf.expand_dims(image, axis=0)  


# Plot the loss function
def plot_loss(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Function Over Epochs')
    plt.legend()
    plt.grid()
    plt.show()

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Function to convert Lab to RGB for visualization
def convert_lab_to_rgb(l_channel, ab_channels):
    """
    Converts L and ab channels into RGB format using skimage's lab2rgb.
    """
    l_channel = (l_channel ) * 100.0  # Scale back to Lab range
    ab_channels = (ab_channels ) * 256.0 - 128 # Scale back to Lab range
    lab = tf.concat([l_channel,ab_channels], axis=-1)
    rgb = tensorflow_io.python.experimental.color_ops.lab_to_rgb(lab)

    return rgb

# Function to predict and visualize colorized images
def predict_and_visualize(model, dataset, num_images=5):
    """
    Predicts and visualizes images from the dataset.
    Args:
        model: Trained colorization model
        dataset: Dataset with grayscale (L channel) inputs
        num_images: Number of images to visualize
    """
    losses = []
    # Take a batch of images
    for batch, ground_truth, l in dataset.take(1):
        # Only take 1 batch for visualization
        # print(batch)
        l_inputs = batch
        l_inputs = l_inputs[:num_images]  # Limit the number of images
        ground_truth = ground_truth[:num_images]

        # Predict ab channels
        predicted_ab = model.predict(l_inputs)
        # predicted_ab = np.array(predicted_ab)
        # Prepare for visualization
        for i in range(num_images):
            l_channel = l_inputs[i]
            
            ab_channels_pred = np.array(predicted_ab[i])

            print(ab_channels_pred.shape)
            # ab_channels_pred = tf.reverse(ab_channels_pred, axis=[-1])

            ab_channels_gt = ground_truth[i]
            # print(tf.reduce_min(ab_channels_gt), tf.reduce_max(ab_channels_gt))
            # print(ground_truth[i].shape, predicted_ab[i].shape, ab_channels_gt.shape, ab_channels_pred.shape)

            # Convert to RGB for visualization
            pred_rgb = convert_lab_to_rgb(l_channel, ab_channels_pred)
            gt_rgb = convert_lab_to_rgb(l_channel, ab_channels_gt)

            # Plot images
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 7, 1)
            plt.imshow(l_channel, cmap="gray")
            plt.title("Grayscale (L Channel)")
            plt.axis("off")

            plt.subplot(1, 7, 2)
            plt.imshow(ab_channels_pred[..., 1], cmap = "gray")
            # plt.imshow(pred_rgb)
            plt.title("Predicted b")
            plt.axis("off")

            plt.subplot(1, 7, 3)
            plt.imshow(ab_channels_gt[..., 1], cmap = "gray")
            # plt.imshow(gt_rgb)
            plt.title("Ground Truth b")
            plt.axis("off")

            plt.subplot(1, 7, 4)
            plt.imshow(ab_channels_pred[..., 0], cmap = "gray")
            # plt.imshow(pred_rgb)
            plt.title("Predicted a")
            plt.axis("off")

            plt.subplot(1, 7, 5)
            plt.imshow(ab_channels_gt[..., 0], cmap = "gray")
            # plt.imshow(gt_rgb)
            plt.title("Ground Truth a")
            plt.axis("off")
            plt.subplot(1, 7, 6)
            # plt.imshow(ab_channels_pred[..., 0], cmap = "gray")
            plt.imshow(pred_rgb)
            plt.title("Predicted rgb")
            plt.axis("off")

            plt.subplot(1, 7, 7)
            # plt.imshow(ab_channels_gt[..., 0], cmap = "gray")
            # print(gt_rgb)
            plt.imshow(gt_rgb)
            plt.title("Ground Truth rgb")
            plt.axis("off")

            plt.show()
            # losses.append(tf.reduce_mean(tf.square( ab_channels_gt, ab_channels_pred)))
            losses.append(tf.reduce_mean(tf.square( pred_rgb, gt_rgb)))
            mse = tf.keras.losses.MeanSquaredError()
            print( mse( ab_channels_gt[..., 1], ab_channels_pred[..., 1]))
            # print(mse( ab_channels_gt, ab_channels_pred))
            print( mse( ab_channels_gt[..., 0], ab_channels_pred[..., 0]))

            # print(tf.reduce_mean(tf.square( ab_channels_gt, ab_channels_pred)) )

        return losses





# Function to convert Lab to RGB for visualization
def convert_lab_to_rgb(l_channel, ab_channels):
    """
    Converts L and ab channels into RGB format using skimage's lab2rgb.
    """
    l_channel = (l_channel ) * 100.0  # Scale back to Lab range
    ab_channels = (ab_channels ) * 256.0 - 128 # Scale back to Lab range
    lab = tf.concat([l_channel,ab_channels], axis=-1)
    rgb = tensorflow_io.python.experimental.color_ops.lab_to_rgb(lab)

    return rgb
# Updated function to include SSIM and PSNR metrics
def predict_lab(model, dataset, out_dir = 'PREDICTIONS', prefix = "LAB"):
    # Take a batch of images
    for batch, ground_truth, l in dataset:
        # Only take 1 batch for visualization
        l_inputs = batch
        # l_inputs = l_inputs[:num_images]  # Limit the number of images
        # ground_truth = ground_truth[:num_images]

        l_inputs = l_inputs  # Limit the number of images
        ground_truth = ground_truth

        # Predict ab channels
        predicted_ab = model.predict(l_inputs)

        # Prepare for visualization and metric calculation
        for i in range(len(l_inputs)):
            l_channel = l_inputs[i]
            ab_channels_pred = np.array(predicted_ab[i])
            ab_channels_gt = ground_truth[i]
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            # Convert to RGB for visualization
            pred_rgb = convert_lab_to_rgb(l_channel, ab_channels_pred)
            img = (np.array(pred_rgb) * 255).astype(np.uint8)  # Convert to 8-bit format
            Image.fromarray(img).save(f"{out_dir}/{prefix}_{timestamp}_{i}.png")



class SaveModelEveryNEpochs(Callback):
    def __init__(self, save_path, interval):
        super(SaveModelEveryNEpochs, self).__init__()
        self.save_path = save_path
        self.interval = interval

    def on_epoch_end(self, epoch, logs=None):
        # Save the model every 'interval' epochs
        if (epoch + 1) % self.interval == 0:
            save_path_with_epoch = f"{self.save_path}.keras"
            self.model.save(save_path_with_epoch)
            print(f"Model saved at {save_path_with_epoch}")
