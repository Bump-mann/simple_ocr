import os

import numpy as np
import tensorflow.keras.backend as K
from PIL import Image
from tensorflow.keras.layers import (Conv2D, Dense, Flatten, Input, Lambda,
                                     MaxPooling2D)
from tensorflow.keras.models import Model

from ..nets.vgg import VGG16


#-------------------------#
#   创建孪生神经网络
#-------------------------#
def siamese(input_shape):
    vgg_model = VGG16()
    
    input_image_1 = Input(shape=input_shape)
    input_image_2 = Input(shape=input_shape)

    #------------------------------------------#
    #   我们将两个输入传入到主干特征提取网络
    #------------------------------------------#
    encoded_image_1 = vgg_model.call(input_image_1)
    encoded_image_2 = vgg_model.call(input_image_2)

    #------------------------------------------#
    #   相减取绝对值，此处算的是l1距离
    #   相当于两个特征向量的距离
    #------------------------------------------#
    l1_distance = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))([encoded_image_1, encoded_image_2])

    #-------------------------#
    #   进行两次全连接
    #-------------------------#
    out = Dense(512,activation='relu')(l1_distance)
    #---------------------------------------------#
    #   利用sigmoid函数将最后的值固定在0-1之间。
    #---------------------------------------------#
    out = Dense(1,activation='sigmoid')(out)

    model = Model([input_image_1, input_image_2], out)
    return model
