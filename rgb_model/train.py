from load_data import load_data
from base_model import create_base_model_perceptual, create_base_model_mse
from model_with_mask import  create_mask_model
from help_functions import train_model
import argparse


def create_rgb_model(image_size, model_type):
    if model_type == "mse":
        return create_base_model_mse(image_size)
    elif model_type == "perc":
        return create_base_model_perceptual(image_size)
    elif model_type == "mask":  
        return create_mask_model(image_size)
    else:
        raise ValueError(f"Invalid model type: {model_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an RGB model.")
    parser.add_argument('model_type', type=str, choices=['mse', 'perc', 'mask'], default='mse', help='Type of model to create (mse, perc, mask)')
    parser.add_argument('data_dir', type=str, default= '../coco_dataset/test2017_small_train/', help='Directory of the dataset')
    parser.add_argument('--image_size', type=int, nargs=2, default=[256, 256], help='Size of the input images (default: [256, 256])')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size for training (default: 5)')
    parser.add_argument('--n_epochs', type=int, default=5, help='Number of epochs for training (default: 5)')
    parser.add_argument('--num_hints', type=int, default=10, help=' num_hints pixels for mask model (default: 10 pixels)')

    args = parser.parse_args()
    data_dir = args.data_dir 
    model_name = "My_RGB_" + args.model_type + "_model"
    train_dataset, val_dataset = load_data(data_dir, args.model_type, args.image_size, args.batch_size, hint_size=args.num_hints)
    base_model = create_rgb_model(args.image_size,  args.model_type)
    train_model(base_model, train_dataset, val_dataset, model_name , model_name + "loss", n_epochs=args.n_epochs)    

