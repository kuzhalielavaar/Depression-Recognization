import tensorflow as tf
from keras import layers, models
import numpy as np
from keras.src.optimizers import Adam
from Classificaltion_Evaluation import ClassificationEvaluation
from sklearn.model_selection import train_test_split


class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsules, capsule_dim, routings=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.routings = routings

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[-1], self.num_capsules * self.capsule_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)

    def call(self, inputs):
        inputs_hat = tf.einsum('...ij,jk->...ik', inputs, self.kernel)
        inputs_hat = tf.reshape(inputs_hat, [-1, inputs.shape[1], self.num_capsules, self.capsule_dim])

        b = tf.zeros(shape=(tf.shape(inputs)[0], tf.shape(inputs)[1], self.num_capsules))

        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=2)
            outputs = self.squash(tf.einsum('...ij,...ijk->...ijk', c, inputs_hat))
            if i < self.routings - 1:
                b += tf.einsum('...ijk,...ijk->...ij', outputs, inputs_hat)

        return tf.reduce_sum(outputs, axis=1)

    def squash(self, s, axis=-1):
        s_squared_norm = tf.reduce_sum(tf.square(s), axis, keepdims=True)
        scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + tf.keras.backend.epsilon())
        return scale * s


# Residual Block for Adaptive Residual Learning
def res_block(x, filters, kernel_size):
    filters = x.shape[-1]
    res = layers.Conv2D(filters, kernel_size, padding='same')(x)
    res = layers.BatchNormalization()(res)
    res = layers.ReLU()(res)

    # Adaptive addition with the original input
    res = layers.Conv2D(filters, kernel_size, padding='same')(res)
    res = layers.Add()([res, x])  # Residual connection
    res = layers.ReLU()(res)
    return res


# MFF-Ada-ResCapsnet Model with three input features
def MFF_Ada_ResCapsNet(input_shape_1, input_shape_2, input_shape_3, classes):
    # Input layers for each feature set
    input_1 = tf.keras.Input(shape=input_shape_1)
    input_2 = tf.keras.Input(shape=input_shape_2)
    input_3 = tf.keras.Input(shape=input_shape_3)

    # Convolution for each input feature
    conv1_1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_1)
    conv1_2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_2)
    conv1_3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_3)

    # Concatenate multiscale features
    fused_features = layers.Concatenate()([conv1_1, conv1_2, conv1_3])

    # Residual block applied to fused features
    res_output = res_block(fused_features, filters=128, kernel_size=(3, 3))

    # Primary Capsules Layer
    primary_capsules = layers.Conv2D(32 * 8, (9, 9), strides=2, padding='valid')(res_output)
    primary_capsules = layers.Reshape(target_shape=[-1, 8])(primary_capsules)

    capsule_layer = CapsuleLayer(num_capsules=10, capsule_dim=16)(primary_capsules)
    flat = layers.Flatten()(capsule_layer)

    output = layers.Dense(classes, activation='sigmoid')(flat)

    model = models.Model([input_1, input_2, input_3], output)
    return model


def Model_MFF_Ada_ResCapsnet(Feature_1, Feature_2, Feature_3, Target, BS=None, sol=None):
    if BS is None:
        BS = 4
    if sol is None:
        sol = [5, 50, 0.01]

    Classes = Target.shape[-1]
    input = (32, 32, 3)

    x_train_1, x_test_1, y_train, y_test = train_test_split(Feature_1, Target, random_state=104, test_size=0.25,
                                       shuffle=True)

    x_train_2, x_test_2, y_train, y_test = train_test_split(Feature_2, Target, random_state=104, test_size=0.25,
                                       shuffle=True)

    x_train_3, x_test_3, y_train, y_test = train_test_split(Feature_3, Target, random_state=104, test_size=0.25,
                                       shuffle=True)

    Train_X_1 = np.resize(x_train_1, (x_train_1.shape[0], input[0], input[1], input[2]))
    Train_X_2 = np.resize(x_train_2, (x_train_2.shape[0], input[0], input[1], input[2]))
    Train_X_3 = np.resize(x_train_3, (x_train_3.shape[0], input[0], input[1], input[2]))

    Test_X_1 = np.resize(x_test_1, (x_test_1.shape[0], input[0], input[1], input[2]))
    Test_X_2 = np.resize(x_test_2, (x_test_2.shape[0], input[0], input[1], input[2]))
    Test_X_3 = np.resize(x_test_3, (x_test_3.shape[0], input[0], input[1], input[2]))

    model = MFF_Ada_ResCapsNet(input_shape_1=input, input_shape_2=input, input_shape_3=input, classes=Classes)
    model.compile(optimizer=Adam(learning_rate=sol[2]), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit([Train_X_1, Train_X_2, Train_X_3], y_train, epochs=int(sol[1]), steps_per_epoch=5, batch_size=BS,
              validation_data=([Test_X_1, Test_X_2, Test_X_3], y_test))
    pred = model.predict([Test_X_1, Test_X_2, Test_X_3])
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    Eval = ClassificationEvaluation(y_test, pred)
    return Eval, pred

