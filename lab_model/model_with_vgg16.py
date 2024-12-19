
import tensorflow as tf
from keras.models import  Model
from keras.layers import Conv2D, UpSampling2D
from help_functions import  psnr, ssim
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Concatenate, BatchNormalization, Flatten, Dense, RepeatVector, Reshape, Lambda
from tensorflow.keras.applications import VGG16


import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate


def create_lab_colorization_model_with_vgg_features(input_shape):
    input_l = Input(shape=(input_shape[0], input_shape[1], 1), name="Input_L")
    input_l_rgb = Lambda(lambda x: tf.image.grayscale_to_rgb(x), name="Pseudo_RGB")(input_l)

    # Extract features from VGG16
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(input_shape[0], input_shape[1], 3))
    vgg_extractor = Model(vgg_model.input, vgg_model.layers[-6].output)  
    vgg_features = vgg_extractor(input_l_rgb)  

    # Process global features with additional layers
    global_features = Conv2D(512, (3, 3), padding='same', strides=(2, 2), activation='relu')(vgg_features)
    global_features = BatchNormalization()(global_features)
    global_features = Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu')(global_features)
    global_features = BatchNormalization()(global_features)
    global_features = UpSampling2D((2, 2))(global_features)

    # Encoder (downsampling) with skip connections
    x1 = Conv2D(64, (3, 3), activation="relu", padding="same", strides=1)(input_l)
    x1 = BatchNormalization()(x1)
    x2 = Conv2D(128, (3, 3), activation="relu", padding="same", strides=2)(x1)  
    x2 = BatchNormalization()(x2)
    x3 = Conv2D(256, (3, 3), activation="relu", padding="same", strides=2)(x2)  
    x3 = BatchNormalization()(x3)
    x4 = Conv2D(512, (3, 3), activation="relu", padding="same", strides=2)(x3)  
    x4 = BatchNormalization()(x4)

    # Merge global features into the decoder
    x4 = Concatenate()([x4, global_features])
    x4 = BatchNormalization()(x4)

    # Decoder (upsampling) with skip connections
    x5 = UpSampling2D((2, 2))(x4)  # Upsampling
    x5 = Concatenate()([x5, x3])  # Skip connection
    x5 = Conv2D(256, (3, 3), activation="relu", padding="same")(x5)
    x5 = BatchNormalization()(x5)

    x6 = UpSampling2D((2, 2))(x5)  # Upsampling
    x6 = Concatenate()([x6, x2])  # Skip connection
    x6 = Conv2D(128, (3, 3), activation="relu", padding="same")(x6)
    x6 = BatchNormalization()(x6)

    x7 = UpSampling2D((2, 2))(x6)  # Upsampling
    x7 = Concatenate()([x7, x1])  # Skip connection
    x7 = Conv2D(64, (3, 3), activation="relu", padding="same")(x7)
    x7 = BatchNormalization()(x7)

    # Final output layer for ab channels
    ab_output = Conv2D(2, (1, 1), activation="sigmoid", name="Output_ab")(x7)

    # Combine inputs and outputs into the final model
    model = Model(inputs=input_l, outputs=[ab_output], name="Lab_Colorization_Model_With_VGG_Features")

    return model



def create_vgg_based_model(image_size , learning_rate = 0.00002):
    input_shape = (image_size[0], image_size[1], 1)
    model = create_lab_colorization_model_with_vgg_features(input_shape)
    model.compile(optimizer=Adam(learning_rate = learning_rate), loss = 'mse', metrics=['mae', 'mse', psnr, ssim])
    model.summary()
    return model