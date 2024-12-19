import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Conv2D, UpSampling2D, InputLayer
from keras.applications import VGG19
from help_functions import  psnr, ssim
from tensorflow.keras.optimizers import Adam





def build_colorization_model(input_shape):
    print(input_shape)
    model = Sequential()
    model.add(InputLayer(shape=input_shape))
    # Encoder
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))

    # Decoder
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))

    return model


vgg = VGG19(include_top=False, weights='imagenet')
vgg.trainable = False  

feature_extractor = Model(inputs=vgg.input, outputs=vgg.get_layer('block5_conv2').output)
def perceptual_loss(y_true, y_pred):
    loss = tf.reduce_mean(tf.square(feature_extractor(y_true) - feature_extractor(y_pred)))
    return loss




def create_base_model_perceptual(image_size , learning_rate = 0.0005):
    input_shape = (image_size[0], image_size[1], 1)
    print(input_shape)
    model = build_colorization_model(input_shape)
    model.compile(optimizer=Adam(learning_rate = learning_rate), loss = perceptual_loss, metrics=['mae', 'mse', psnr, ssim])
    model.summary()
    return model


def create_base_model_mse(image_size, learning_rate = 0.0005):
    input_shape = (image_size[0], image_size[1], 1)
    model = build_colorization_model(input_shape)
    model.compile(optimizer=Adam(learning_rate = learning_rate), loss = 'mse', metrics=['mae', 'mse', psnr, ssim])
    model.summary()
    return model
