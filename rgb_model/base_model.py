from keras.models import Sequential, Model
from keras.layers import Conv2D, UpSampling2D, InputLayer, BatchNormalization, Input
from keras.applications import VGG19
from keras.callbacks import Callback


def build_colorization_model(input_shape):
    model = Sequential()
    model.add(InputLayer(shape=input_shape))
    print(input_shape)
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




def create_base_model(image_size):
    input_shape = (image_size[0], image_size[1], 1)  # Grayscale images have 1 channel
    model = build_colorization_model(input_shape)
    # model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.compile(optimizer='adam', loss = perceptual_loss, metrics=['mae'])
    model.summary()
    return model
    
def fit_base_model(model, train_dataset, val_dataset, save_path, n_epochs = 10, save_interval = 10):
    save_callback = SaveModelEveryNEpochs(save_path=save_path, interval=save_interval)
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=n_epochs,
        callbacks=[save_callback]
    )
    return history