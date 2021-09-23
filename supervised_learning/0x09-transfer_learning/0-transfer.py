#!/usr/bin/env python3
"""
trains a convolutional neural network
to classify the CIFAR 10 dataset
"""
import tensorflow.keras as K


def preprocess_data(X, Y):
    """pre-processes the data for your model"""
    X_p = K.applications.densenet.preprocess_input(
        X, data_format="channels_last")
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


if __name__ == '__main__':
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    shape = (32, 32, 3)
    ins = K.Input(shape=shape)
    resized_ins = K.layers.Lambda(lambda x: K.backend.resize_images(x, height_factor=(
        224 // 32), width_factor=(224 // 32), data_format="channels_last"))(ins)

    DenseNet121 = K.applications.DenseNet121(
        include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    act = K.activations.relu

    X = K.layers.Dropout(0.2)(K.layers.Dense(
        500, activation=act)(K.layers.Flatten()(DenseNet121(resized_ins, training=False))))
    outs = K.layers.Dense(10, activation='softmax')(X)
    model = K.Model(inputs=ins, outputs=outs)

    DenseNet121.trainable = False

    model.compile(loss='categorical_crossentropy',
                  optimizer=K.optimizers.Adam(), metrics=['accuracy'])

    history = model.fit(x=X_train, y=Y_train, validation_data=(
        X_test, Y_test), batch_size=300, epochs=5, verbose=True)

    model.save('cifar10.h5')
