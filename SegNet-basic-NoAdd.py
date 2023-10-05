from __future__ import absolute_import
from __future__ import print_function


import keras.models as models
from layers import MaxPoolingWithArgmax2D
from layers import MaxUnpooling2D
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Merge, Permute
from keras.layers.convolutional import Convolution2D, Input, Model
from keras.layers.normalization import BatchNormalization


from keras import backend as K

import numpy as np
import json
np.random.seed(7) # 0bserver07 for reproducibility

data_shape = 360*480
n_labels = 12

def create_segnet(input_shape,
                  n_labels,
                  num_filters=32,
                  output_mode="softmax"):
    inputs = Input(shape=input_shape)

    conv_1 = Convolution2D(num_filters, (7, 7), padding="same", kernel_initializer='he_normal')(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)

    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv_1)

    conv_1 = Convolution2D(num_filters, (7, 7), padding="same", kernel_initializer='he_normal')(pool_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)

    pool_1, mask_2 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv_1)

    conv_1 = Convolution2D(num_filters, (7, 7), padding="same", kernel_initializer='he_normal')(pool_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)

    pool_1, mask_3 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv_1)

    conv_1 = Convolution2D(2 * num_filters, (7, 7), padding="same", kernel_initializer='he_normal')(pool_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)

    pool_2, mask_4 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv_1)

    # Decoding

    unpool_1 = MaxUnpooling2D(pool_size=(2, 2))([pool_2, mask_4])

    conv_2 = Convolution2D(num_filters, (7, 7), padding="same", kernel_initializer='he_normal')(unpool_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation("relu")(conv_2)

    unpool_2 = MaxUnpooling2D(pool_size=(2, 2))([conv_2, mask_3])

    conv_2 = Convolution2D(n_labels, (1, 1), padding="same", kernel_initializer='he_normal')(unpool_2)
    conv_2 = BatchNormalization()(conv_2)
    outputs = Activation(output_mode)(conv_2)

    unpool_2 = MaxUnpooling2D(pool_size=(2, 2))([conv_2, mask_2])

    conv_2 = Convolution2D(n_labels, (1, 1), padding="same", kernel_initializer='he_normal')(unpool_2)
    conv_2 = BatchNormalization()(conv_2)
    outputs = Activation(output_mode)(conv_2)

    unpool_2 = MaxUnpooling2D(pool_size=(2, 2))([conv_2, mask_1])

    conv_2 = Convolution2D(n_labels, (1, 1), padding="same", kernel_initializer='he_normal')(unpool_2)
    conv_2 = BatchNormalization()(conv_2)
    outputs = Activation(output_mode)(conv_2)

    segnet = Model(inputs=inputs, outputs=outputs)
    return segnet


segnet_basic = models.Sequential()

segnet_basic = create_segnet(data_shape, n_labels)

# Save model to JSON

with open('segNet_basic_model.json', 'w') as outfile:
    outfile.write(json.dumps(json.loads(segnet_basic.to_json()), indent=2))