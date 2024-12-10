from keras.layers import Concatenate, Input
from keras.models import Sequential, Model
from keras.layers import Conv2D, UpSampling2D, InputLayer, BatchNormalization, Input
from keras.applications import VGG19
def build_colorization_model_with_mask(input_shape):
    # Inputs
    grayscale_input = Input(shape=input_shape, name="grayscale_input")  # (128, 128, 1)
    mask_input = Input(shape=(input_shape[0], input_shape[1], 3), name="mask_input")  # (128, 128, 3)

    # Concatenate grayscale and mask
    x = Concatenate()([grayscale_input, mask_input])  # (128, 128, 4)

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



def create_mask_model():
    input_shape = (image_size[0], image_size[1], 1)  
    model_mask = build_colorization_model_with_mask(input_shape)
    model_mask.summary()
    model_mask.compile(optimizer='adam', loss= 'mse', metrics=['mae'])
    return model_mask
    

def fit_mask_model(model, train_dataset, val_dataset, save_path, n_epochs = 10, save_interval = 10):
    
    save_callback = SaveModelEveryNEpochs(save_path=save_path, interval=save_interval)
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=n_epochs,
        callbacks=[save_callback]
    )
    return history