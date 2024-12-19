import argparse
from tensorflow.keras.models import load_model
from help_functions import  psnr, ssim, predict_rgb
from load_data import load_inference_data

def load_model_from_path(model_path, model_type, image_size):
    custom_objects = {
    "psnr": psnr,
    "ssim": ssim,
        }
    model = load_model(model_path, custom_objects = custom_objects)
    return model 


if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser(description="Colorise with model.")
    parser.add_argument('model_type', type=str, choices=['mse', 'perc', "mask"], default='vgg_based', help='Type of model to create (base, vgg_based)')
    parser.add_argument('data_dir', type=str, default= 'coco_dataset/test2017_small_train/', help='Directory of the dataset')
    parser.add_argument('model_path', type=str, default='', help='DOWNLOADED_MODELS/lab_model_256x256_75_epochs_mse_10000imgs_benchmark.keras')   
    parser.add_argument('--out_dir', type=str, default='PREDICTIONS', help=' OUT_DIR for predictions (default: PREDICTIONS)')

    args = parser.parse_args()
    model_path = args.model_path
    image_size = (256, 256)
    batch_size = 8
    model = load_model_from_path(model_path, args.model_type, image_size )
    inference_dataset = load_inference_data(args.data_dir, image_size, batch_size)
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)  
    predict_rgb(model, inference_dataset, out_dir=args.out_dir, prefix= "RGB_" + args.model_type)