from load_data import load_data
from base_model import create_lab_base_model
from model_with_vgg16 import  create_vgg_based_model
from help_functions import train_model
import argparse


def create_lab_model(image_size, model_type):
    if model_type == "base":
        return create_lab_base_model(image_size)
    elif model_type == "vgg_based":
        return create_vgg_based_model(image_size)
    else:
        raise ValueError(f"Invalid model type: {model_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an RGB model.")
    parser.add_argument('model_type', type=str, choices=['base', 'vgg_based'], default='mse', help='Type of model to create (mse, perc, mask)')
    parser.add_argument('data_dir', type=str, default= '../coco_dataset/test2017_small_train/', help='Directory of the dataset')
    parser.add_argument('--image_size', type=int, nargs=2, default=[256, 256], help='Size of the input images (default: [256, 256])')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size for training (default: 5)')
    parser.add_argument('--n_epochs', type=int, default=5, help='Number of epochs for training (default: 5)')

    args = parser.parse_args()
    data_dir = args.data_dir 
    model_name = "My_LAB_" + args.model_type + "_model"
    train_dataset, val_dataset = load_data(data_dir, args.image_size, args.batch_size)
    base_model = create_lab_model(args.image_size,  args.model_type)
    train_model(base_model, train_dataset, val_dataset, model_name , model_name + "loss", n_epochs=args.n_epochs)    

