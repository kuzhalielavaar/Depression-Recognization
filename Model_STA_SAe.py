import keras
import numpy as np
from sklearn.model_selection import train_test_split
from keras import layers, regularizers, Model


# https://www.geeksforgeeks.org/sparse-autoencoders-in-deep-learning/
# https://github.com/sudharsan13296/Hands-On-Deep-Learning-Algorithms-with-Python/blob/master/10.%20Reconsturcting%20Inputs%20using%20Autoencoders/10.09%20Building%20the%20Sparse%20Autoencoder.ipynb


# Spatial Attention Block
def spatial_attention(x):
    avg_pool = layers.GlobalAveragePooling2D()(x)
    max_pool = layers.GlobalMaxPooling2D()(x)

    # Concatenate both along spatial dimensions
    concat = layers.Concatenate()([avg_pool, max_pool])
    dense1 = layers.Dense(x.shape[-1] // 8, activation='relu')(concat)
    dense2 = layers.Dense(x.shape[-1], activation='sigmoid')(dense1)
    return layers.Multiply()([x, dense2])


# Temporal Attention Block (Simulating for temporal, assuming input is time-series like or sequential data)
def temporal_attention(x):
    avg_pool = layers.GlobalAveragePooling2D()(x)
    max_pool = layers.GlobalMaxPooling2D()(x)

    concat = layers.Concatenate()([avg_pool, max_pool])
    dense1 = layers.Dense(x.shape[-1] // 8, activation='relu')(concat)
    dense2 = layers.Dense(x.shape[-1], activation='sigmoid')(dense1)
    return layers.Multiply()([x, dense2])


# Sparse Autoencoder (with attention)
def Model_STA_SAe(Data, Targets):
    input_img = layers.Input(shape=(32, 32, 1))  # Assuming the input is 28x28 images

    IMG_SIZE = 32
    Datas = np.zeros((Data.shape[0], IMG_SIZE, IMG_SIZE, 1))
    for i in range(Data.shape[0]):
        temp = np.resize(Data[i], (IMG_SIZE, IMG_SIZE, 1))
        Datas[i] = temp

    train_data, test_data, train_target, test_target = train_test_split(Datas, Targets, random_state=104,
                                                                        test_size=0.25, shuffle=True)

    # Encoder
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
    x = spatial_attention(x)  # Spatial Attention Block
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = temporal_attention(x)  # Temporal Attention Block
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same", activity_regularizer=regularizers.l1(10e-5))(
        x)  # Sparsity regularization

    # Latent space representation (compressed version of the input)
    encoded = layers.MaxPooling2D((2, 2), padding="same")(x)

    # Decoder
    x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation="relu", padding="same")(encoded)
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
    decoded = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

    # Autoencoder Model (STA-SAe)
    model = Model(input_img, decoded)
    model.compile(optimizer='sgd', loss='mse', metrics=[keras.metrics.SparseCategoricalCrossentropy()])
    model.summary()

    # Train the model
    model.fit(x=train_data, y=train_data, epochs=10, batch_size=4, shuffle=True, validation_data=(test_data, test_data))
    predictions = model.predict(test_data)

    layerNo = -3
    intermediate_model = Model(inputs=model.input, outputs=model.layers[layerNo].output)
    Feats = intermediate_model.predict(np.concatenate((train_data, test_data)))

    # Reshape Features
    Feats = np.asarray(Feats)
    Feature = np.resize(Feats, (Feats.shape[0], Feats.shape[-1]))
    return Feature

