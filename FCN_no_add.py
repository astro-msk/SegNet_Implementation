from __future__ import absolute_import
from __future__ import print_function


import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Merge, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Input, Model
from keras.layers.normalization import BatchNormalization


from keras import backend as K

import numpy as np
import json
np.random.seed(7) # 0bserver07 for reproducibility

data_shape = 360*480
n_labels = 12

def create_fcnnoadd(input_shape,
                  n_labels,
                  num_filters=32,
                  output_mode="softmax"):
    inputs = Input(shape=input_shape)

    conv_1 = Convolution2D(num_filters, (8, 8), padding="same", kernel_initializer='he_normal')(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)

    pool_1 = MaxPooling2D(pool_size=(2,2))(conv_1)

    conv_1 = Convolution2D(2 * num_filters, (8, 8), padding="same", kernel_initializer='he_normal')(pool_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)

    pool_1 = MaxPooling2D(pool_size=(2,2))(conv_1)

    conv_1 = Convolution2D(num_filters, (8, 8), padding="same", kernel_initializer='he_normal')(pool_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)

    pool_1 = MaxPooling2D(pool_size=(2,2))(conv_1)

    conv_1 = Convolution2D(num_filters, (8, 8), padding="same", kernel_initializer='he_normal')(pool_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)

    pool_2 = MaxPooling2D(pool_size=(2,2))(conv_1)

    #Decoding

    unpool_1 = UpSampling2D(pool_size=(2,2))(pool_2)

    conv_3 = Convolution2D(n_labels, (8, 8), padding="same", kernel_initializer='he_normal')(unpool_1)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)

    unpool_1 = UpSampling2D(pool_size=(2,2))(conv_3)

    conv_3 = Convolution2D(n_labels, (1, 1), padding="same", kernel_initializer='he_normal')(unpool_1)
    conv_3 = BatchNormalization()(conv_3)
    outputs = Activation(output_mode)(conv_3)

    unpool_1 = UpSampling2D(pool_size=(2,2))(pool_2)

    conv_3 = Convolution2D(n_labels, (8, 8), padding="same", kernel_initializer='he_normal')(unpool_1)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)

    unpool_1 = UpSampling2D(pool_size=(2,2))(pool_2)

    conv_3 = Convolution2D(n_labels, (8, 8), padding="same", kernel_initializer='he_normal')(unpool_1)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)

    fcnNoAdd = Model(inputs=inputs, outputs=outputs)
    return fcnNoAdd


fcnNoAdd_basic = models.Sequential()

fcnNoAdd_basic = create_fcnnoadd(data_shape, n_labels)# input layer

# Save model to JSON

with open('fcnNoAdd_basic_model.json', 'w') as outfile:
    outfile.write(json.dumps(json.loads(fcnNoAdd_basic.to_json()), indent=2))