

import tensorflow as tf
from skimage.color import rgb2gray, rgb2lab
from skimage.restoration import estimate_sigma
from pathlib import Path

from load_data import image_size, batch_size
# # Ajouter des optimisations (mélange, préchargement)
# train_dataset = train_dataset.shuffle(buffer_size=1000).prefetch(tf.data.experimental.AUTOTUNE)
# val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)


def evaluate_model_on_validation(model, val_data):
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


# NIQE Calculation (Placeholder Example)
def calculate_niqe(image):
    """
    Calculate the NIQE score for an image.
    A placeholder logic using noise sigma as a proxy for demonstration.
    In practice, use a pre-trained NIQE model.
    """
    gray_image = rgb2gray(image)
    sigma = estimate_sigma(gray_image, average_sigmas=True)
    niqe_score = sigma
    return niqe_score

def calculate_colorfulness(image):
    """
    Calculate the colorfulness metric for an RGB image.
    """
    (R, G, B) = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    std_rg, mean_rg = np.std(rg), np.mean(rg)
    std_yb, mean_yb = np.std(yb), np.mean(yb)
    colorfulness = np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)
    return colorfulness

def calculate_brightness_contrast(image):
    """
    Calculate brightness and contrast for an image.
    Brightness is the mean luminance, and contrast is the standard deviation.
    """
    luminance = 0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]
    brightness = np.mean(luminance)
    contrast = np.std(luminance)

    return brightness, contrast

