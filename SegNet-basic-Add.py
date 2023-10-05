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

def create_segnet_Add(input_shape,
                  n_labels,
                  num_filters=32,
                  output_mode="softmax"):
    inputs = Input(shape=input_shape)

    conv_1 = Convolution2D(num_filters, (7, 7), padding="same", kernel_initializer='he_normal')(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)

    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv_1)

    conv_2 = Convolution2D(2 * num_filters, (7, 7), padding="same", kernel_initializer='he_normal')(pool_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation("relu")(conv_2)

    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv_2)

    conv_3 = Convolution2D(num_filters, (7, 7), padding="same", kernel_initializer='he_normal')(pool_2)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)

    pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv_1)

    conv_4 = Convolution2D(2 * num_filters, (7, 7), padding="same", kernel_initializer='he_normal')(pool_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation("relu")(conv_4)

    pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv_1)

    #Decoding Layers

    adj_lay1 = Convolution2D(2 * num_filters, (7, 7), padding="same", kernel_initializer='he_normal')(0.01*conv_4)
    upool_1 = MaxUnpooling2D(pool_size=(2,2))([pool_4, mask_4])
    
    unpool_1 = upool_1 + adj_lay1[:, :, 5: (5 + upool_1.size()[2]), 5: (5 + upool_1.size()[3])]

    conv_5 = Convolution2D(num_filters, (7, 7), padding="same", kernel_initializer='he_normal')(unpool_1)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)

    adj_lay2 = Convolution2D(2 * num_filters, (7, 7), padding="same", kernel_initializer='he_normal')(0.01*conv_3)
    upool_2 = MaxUnpooling2D(pool_size=(2,2))([conv_5, mask_3])
    
    unpool_2 = upool_2 + adj_lay2[:, :, 5: (5 + upool_2.size()[2]), 5: (5 + upool_2.size()[3])]

    conv_6 = Convolution2D(num_filters, (7, 7), padding="same", kernel_initializer='he_normal')(unpool_2)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)

    adj_lay3 = Convolution2D(2 * num_filters, (7, 7), padding="same", kernel_initializer='he_normal')(0.01*conv_2)
    upool_3 = MaxUnpooling2D(pool_size=(2,2))([conv_6, mask_2])
    
    unpool_3 = upool_3 + adj_lay3[:, :, 5: (5 + upool_3.size()[2]), 5: (5 + upool_3.size()[3])]

    conv_7= Convolution2D(num_filters, (7, 7), padding="same", kernel_initializer='he_normal')(unpool_3)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)

    adj_lay4 = Convolution2D(num_filters, (7, 7), padding="same", kernel_initializer='he_normal')(0.01*conv_1)
    upool_4 = MaxUnpooling2D(pool_size=(2,2))([conv_7, mask_1])
    
    unpool_4 = upool_4 + adj_lay4[:, :, 5: (5 + upool_4.size()[2]), 5: (5 + upool_4.size()[3])]

    conv_8 = Convolution2D(n_labels, (1, 1), padding="same", kernel_initializer='he_normal')(unpool_4)
    conv_8 = BatchNormalization()(conv_8)
    outputs = Activation(output_mode)(conv_8)

    segnet_Add_basic = Model(inputs=inputs, outputs=outputs)
    return segnet_Add_basic


segnet_Add_basic = models.Sequential()

segnet_Add_basic = create_segnet_Add(data_shape, n_labels)

# Save model to JSON

with open('segNet_Add_basic_model.json', 'w') as outfile:
    outfile.write(json.dumps(json.loads(segnet_Add_basic.to_json()), indent=2))