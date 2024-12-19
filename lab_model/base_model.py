import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Conv2D, UpSampling2D, InputLayer
from keras.applications import VGG19
from help_functions import  psnr, ssim
from tensorflow.keras.optimizers import Adam



import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, concatenate, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate


def create_lab_colorization_model_with_batch_norm(input_shape):
    input_l = Input(shape=(input_shape[0], input_shape[1], 1), name="Input_L")

    # Encoder 
    x1 = Conv2D(64, (3, 3), padding="same", strides=1)(input_l)
    x1 = BatchNormalization()(x1) 
    x1 = tf.keras.layers.Activation("relu")(x1)

    x2 = Conv2D(128, (3, 3), padding="same", strides=2)(x1)  
    x2 = BatchNormalization()(x2)  
    x2 = tf.keras.layers.Activation("relu")(x2)

    x3 = Conv2D(256, (3, 3), padding="same", strides=2)(x2)  
    x3 = BatchNormalization()(x3)  
    x3 = tf.keras.layers.Activation("relu")(x3)

    x4 = Conv2D(512, (3, 3), padding="same", strides=2)(x3) 
    x4 = BatchNormalization()(x4)  
    x4 = tf.keras.layers.Activation("relu")(x4)

    # Decoder (upsampling)
    x5 = UpSampling2D((2, 2))(x4) 
    x5 = Concatenate()([x5, x3]) 
    x5 = Conv2D(256, (3, 3), padding="same")(x5)
    x5 = BatchNormalization()(x5) 
    x5 = tf.keras.layers.Activation("relu")(x5)

    x6 = UpSampling2D((2, 2))(x5) 
    x6 = Concatenate()([x6, x2]) 
    x6 = Conv2D(128, (3, 3), padding="same")(x6)
    x6 = BatchNormalization()(x6) 
    x6 = tf.keras.layers.Activation("relu")(x6)

    x7 = UpSampling2D((2, 2))(x6)  
    x7 = Concatenate()([x7, x1])  
    x7 = Conv2D(64, (3, 3), padding="same")(x7)
    x7 = BatchNormalization()(x7)  
    x7 = tf.keras.layers.Activation("relu")(x7)

    ab_output = Conv2D(2, (1, 1), activation="sigmoid", name="Output_ab")(x7)
    model = Model(inputs=input_l, outputs=ab_output, name="Lab_Colorization_Model_With_Batch_Norm")
    return model


def create_lab_base_model(image_size , learning_rate = 0.00002):
    input_shape = (image_size[0], image_size[1], 1)
    model = create_lab_colorization_model_with_batch_norm(input_shape)
    model.compile(optimizer=Adam(learning_rate = learning_rate), loss = 'mse', metrics=['mae', 'mse', psnr, ssim])
    model.summary()
    return model