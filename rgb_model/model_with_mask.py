from keras.layers import Concatenate, Input
from keras.models import Sequential, Model
from keras.layers import Conv2D, UpSampling2D, Input
from help_functions import SaveModelEveryNEpochs, psnr, ssim
from tensorflow.keras.optimizers import Adam
def build_colorization_model_with_mask(input_shape):
    # Inputs
    grayscale_input = Input(shape=input_shape, name="grayscale_input")  
    mask_input = Input(shape=(input_shape[0], input_shape[1], 3), name="mask_input") 

    # Concatenate grayscale and mask
    x = Concatenate()([grayscale_input, mask_input])  

    # Encoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', strides=2)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)

    # Decoder
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    output = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)  # Final colorized output

    model = Model(inputs=[grayscale_input, mask_input], outputs=output)
    return model



def create_mask_model( image_size):
    input_shape = (image_size[0], image_size[1], 1)  
    model_mask = build_colorization_model_with_mask(input_shape)
    model_mask.summary()
    model_mask.compile(optimizer='adam', loss= 'mse', metrics=['mae'])
    return model_mask
    

def create_mask_model(image_size , learning_rate = 0.0005):
    input_shape = (image_size[0], image_size[1], 1)
    model = build_colorization_model_with_mask(input_shape)
    model.compile(optimizer=Adam(learning_rate = learning_rate), loss = 'mse', metrics=['mae', 'mse', psnr, ssim])
    model.summary()
    return model

