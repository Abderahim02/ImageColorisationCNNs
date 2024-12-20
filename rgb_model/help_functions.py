import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt
from keras.callbacks import Callback
import json


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


def display_from_dataset(dataset, num_images=5):
    plt.figure(figsize=(10, num_images * 3))
    count = 0
    for grayscale_batch, rgb_batch in dataset.take(1): 
        for i in range(num_images):
            if count >= len(grayscale_batch):  # Stop if fewer images than num_images in batch
                break
            grayscale_image = grayscale_batch[i].numpy().squeeze()  
            rgb_image = rgb_batch[i].numpy() 

            # Plot grayscale image
            plt.subplot(num_images, 2, 2 * count + 1)
            plt.imshow(grayscale_image, cmap='gray')
            plt.title("Grayscale Image")
            plt.axis("off")
            # Plot RGB image
            plt.subplot(num_images, 2, 2 * count + 2)
            plt.imshow(rgb_image)
            plt.title("RGB Image")
            plt.axis("off")

            count += 1
            if count >= num_images:
                break
    plt.show()

# display_from_dataset(train_dataset, num_images=5)



# Charger une image grayscale pour prédiction
def load_grayscale_image(image_path, target_size):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=1)  
    image = tf.image.resize(image, target_size) / 255.0  
    return tf.expand_dims(image, axis=0)  


def predict_and_visualize_rgb(model, data):
    for grayscale_batch, rgb_batch in data.take(1):
        predicted_rgb = model.predict(grayscale_batch)
        # Initialize PSNR and SSIM lists
        psnr_scores = []
        ssim_scores = []
        plt.figure(figsize=(15, 10))
        for i in range(5):  # Display 5 examples
            true_image = rgb_batch[i].numpy()
            pred_image = np.clip(predicted_rgb[i], 0, 1)  # Ensure predictions are in [0, 1]
            psnr = peak_signal_noise_ratio(true_image, pred_image, data_range=1)
            ssim = structural_similarity(
                true_image, 
                pred_image, 
                channel_axis=-1, 
                data_range=1, 
                win_size=3  
            )
            psnr_scores.append(psnr)
            ssim_scores.append(ssim)

            plt.subplot(4, 5, i + 1)
            plt.title("Grayscale")
            plt.imshow(grayscale_batch[i].numpy().squeeze(), cmap='gray')
            plt.axis('off')

            plt.subplot(4, 5, i + 6)
            plt.title("True RGB")
            plt.imshow(true_image)
            plt.axis('off')

            plt.subplot(4, 5, i + 11)
            plt.title("Predicted RGB")
            plt.imshow(pred_image)
            plt.axis('off')

            plt.subplot(4, 5, i + 16)
            plt.title(f"PSNR: {psnr:.2f}\nSSIM: {ssim:.2f}")
            plt.axis('off')

        plt.tight_layout()
        plt.show()
        print(f"Average PSNR: {np.mean(psnr_scores):.2f}")
        print(f"Average SSIM: {np.mean(ssim_scores):.2f}")


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



def display_sample_images_with_mask(dataset, num_samples=5):
    """Displays a few examples of grayscale images, masks, and original images."""
    for (grayscale, mask), original in dataset.take(num_samples):
        for i in range(min(num_samples, grayscale.shape[0])):  # Handle batch
            # Convert tensors to numpy arrays
            grayscale_img = grayscale[i].numpy().squeeze()  # (128, 128, 1) -> (128, 128)
            mask_img = mask[i].numpy()
            original_img = original[i].numpy()

            # Plot the images
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.title("Grayscale")
            plt.imshow(grayscale_img, cmap='gray')
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.title("Mask")
            plt.imshow(mask_img)
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.title("Original")
            plt.imshow(original_img)
            plt.axis('off')

            plt.tight_layout()
            plt.show()

# Call the function on the dataset
def predict_with_mask(model, grayscale_image, mask):
    grayscale_image = np.expand_dims(grayscale_image, axis=0)  
    mask = np.expand_dims(mask, axis=0)  
    prediction = model.predict([grayscale_image, mask])
    return prediction[0]  



def predict_and_visualize(model, dataset):
    """Predicts and visualizes the results on a batch of data."""
    for (grayscale_batch, mask_batch), true_rgb_batch in dataset.take(1):
        predicted_rgb_batch = model.predict([grayscale_batch, mask_batch])

        psnr_scores = []
        ssim_scores = []

        plt.figure(figsize=(15, 10))

        # Display 5 examples
        for i in range(min(5, grayscale_batch.shape[0])):
            grayscale_image = grayscale_batch[i].numpy().squeeze()  # (128, 128, 1) -> (128, 128)
            mask_image = mask_batch[i].numpy()  # (128, 128, 3)
            true_rgb_image = true_rgb_batch[i].numpy()  # (128, 128, 3)
            predicted_rgb_image = np.clip(predicted_rgb_batch[i], 0, 1)  # (128, 128, 3)

            # Compute metrics
            psnr = peak_signal_noise_ratio(true_rgb_image, predicted_rgb_image, data_range=1)
            ssim = structural_similarity(
                true_rgb_image,
                predicted_rgb_image,
                channel_axis=-1,
                data_range=1,
                win_size=3
            )
            psnr_scores.append(psnr)
            ssim_scores.append(ssim)

            plt.subplot(5, 5, i + 1)
            plt.title("Grayscale")
            plt.imshow(grayscale_image, cmap='gray')
            plt.axis('off')

            plt.subplot(5, 5, i + 6)
            plt.title("Hint Mask")
            plt.imshow(mask_image)
            plt.axis('off')

            plt.subplot(5, 5, i + 11)
            plt.title("True RGB")
            plt.imshow(true_rgb_image)
            plt.axis('off')

            plt.subplot(5, 5, i + 16)
            plt.title("Predicted RGB")
            plt.imshow(predicted_rgb_image)
            plt.axis('off')

            plt.subplot(5, 5, i + 21)
            plt.title(f"PSNR: {psnr:.2f}\nSSIM: {ssim:.2f}")
            plt.axis('off')

        plt.tight_layout()
        plt.show()
        print(f"Average PSNR: {np.mean(psnr_scores):.2f}")
        print(f"Average SSIM: {np.mean(ssim_scores):.2f}")
        break  


def evaluate_mask_model_on_validation(model, val_data):
    psnr_scores = []
    ssim_scores = []

    # Iterate over the entire validation dataset
    for (grayscale_batch, mask_batch), rgb_batch in val_data:
        # Predict using the model
        predicted_batch = model.predict([grayscale_batch, mask_batch])

        # Iterate through the batch
        for i in range(grayscale_batch.shape[0]):
            true_image = rgb_batch[i].numpy()  # Ground truth
            pred_image = np.clip(predicted_batch[i], 0, 1)  # Ensure predictions are in [0, 1]

            # Compute PSNR and SSIM
            psnr = peak_signal_noise_ratio(true_image, pred_image, data_range=1)
            ssim = structural_similarity(
                true_image,
                pred_image,
                channel_axis=-1,
                data_range=1,
                win_size=3
            )

            # Append scores
            psnr_scores.append(psnr)
            ssim_scores.append(ssim)

    # Compute mean scores
    mean_psnr = np.mean(psnr_scores)
    mean_ssim = np.mean(ssim_scores)

    print(f"Mean PSNR: {mean_psnr:.2f}")
    print(f"Mean SSIM: {mean_ssim:.2f}")

    return mean_psnr, mean_ssim




def evaluate_base_model_on_validation(model, val_data):
    psnr_scores = []
    ssim_scores = []

    # Iterate over the entire validation dataset
    for grayscale_batch, rgb_batch in val_data:
        predicted_batch = model.predict(grayscale_batch)
        # Iterate through the batch
        for i in range(grayscale_batch.shape[0]):
            true_image = rgb_batch[i].numpy()
            pred_image = np.clip(predicted_batch[i], 0, 1)
            psnr = peak_signal_noise_ratio(true_image, pred_image, data_range=1)
            ssim = structural_similarity(
                true_image,
                pred_image,
                channel_axis=-1,
                data_range=1,
                win_size=3
            )

            psnr_scores.append(psnr)
            ssim_scores.append(ssim)
    mean_psnr = np.mean(psnr_scores)
    mean_ssim = np.mean(ssim_scores)
    print(f"Mean PSNR: {mean_psnr:.2f}")
    print(f"Mean SSIM: {mean_ssim:.2f}")

    return mean_psnr, mean_ssim


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


def predict_rgb(model, data, prefix="RGB", out_dir="PREDICTIONS"):
    for grayscale_batch, rgb_batch in data:
        predicted_rgb = model.predict(grayscale_batch)

        for i in range(grayscale_batch.shape[0]):  
            true_image = rgb_batch[i].numpy()
            pred_image = np.clip(predicted_rgb[i], 0, 1)  # Ensure predictions are in [0, 1]
            img = (pred_image * 255).astype(np.uint8)  # Convert to 8-bit format
            time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            Image.fromarray(img).save(f"{out_dir}/{prefix}_{time_stamp}_{i}.png")