from load_data import load_data
from base_model import create_base_model, fit_base_model
from model_with_mask import  create_mask_model, fit_mask_model

from help_functions import evaluate_base_model_on_validation, evaluate_mask_model_on_validation
if __name__ == "__main__":
    data_dir = 'coco_dataset/test2017/'
    train_dataset, val_dataset = load_data(data_dir)
    image_size = (256, 256)
    batch_size = 8
    base_model = create_base_model(data_dir, image_size)

    history = fit_base_model(base_model)



