'''


图片相似度对比 适用于图标点选
'''


import os

import cv2
import tensorflow as tf
import numpy as np
from PIL import Image
from  .nets.siamese import siamese
from  .utils.utils import letterbox_image, preprocess_input, cvtColor, show_config

# -----------nets----------------------------------------#
#   使用自己训练好的模型预测需要修改model_path参数
# ---------------------------------------------------#
class Siamese(object):
    _defaults = {
        # -----------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path
        #   model_path指向logs文件夹下的权值文件
        # -----------------------------------------------------#
        "model_path": './models/图标点选_相似度.h5',
        # -----------------------------------------------------#
        #   输入图片的大小。
        # -----------------------------------------------------#
        "input_shape": [60, 60],
        # --------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize
        #   否则对图像进行CenterCrop
        # --------------------------------------------------------------------#
        "letterbox_image": True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化Siamese
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        self.generate()

        show_config(**self._defaults)

    # ---------------------------------------------------#
    #   载入模型
    # ---------------------------------------------------#
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        # ---------------------------#
        #   载入模型与权值
        # ---------------------------#
        self.model = siamese([self.input_shape[0], self.input_shape[1], 3])
        self.model.load_weights(self.model_path)
        print('{} model loaded.'.format(model_path))

    @tf.function
    def get_pred(self, photo):
        preds = self.model(photo, training=False)
        return preds

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image_1, image_2):
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        # ---------------------------------------------------------#
        image_1 = cvtColor(image_1)
        image_2 = cvtColor(image_2)

        # ---------------------------------------------------#
        #   对输入图像进行不失真的resize
        # ---------------------------------------------------#
        image_1 = letterbox_image(image_1, [self.input_shape[1], self.input_shape[0]], self.letterbox_image)
        image_2 = letterbox_image(image_2, [self.input_shape[1], self.input_shape[0]], self.letterbox_image)

        # ---------------------------------------------------------#
        #   归一化+添加上batch_s  ize维度
        # ---------------------------------------------------------#
        photo1 = np.expand_dims(preprocess_input(np.array(image_1, np.float32)), 0)
        photo2 = np.expand_dims(preprocess_input(np.array(image_2, np.float32)), 0)

        # ---------------------------------------------------#
        #   获得预测结果，output输出为概率
        # ---------------------------------------------------#
        output = np.array(self.get_pred([photo1, photo2])[0])

        # plt.subplot(1, 2, 1)
        # plt.imshow(np.array(image_1))
        #
        # plt.subplot(1, 2, 2)
        # plt.imshow(np.array(image_2))
        # plt.text(-12, -12, 'Similarity:%.3f' % output, ha='center', va='bottom', fontsize=11)
        # plt.show()
        return output




'''
以下是测试代码 (本来想着在每个代码下面加测试来着，但是认为不好就废弃掉了)
'''

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    model = Siamese()



    for i in range(1,6):
        image_1 = Image.open('../test/图标点选/背景图__切割后图片_{}.png'.format(i))
        max = 0
        for j in range(1,4):


            image_2 = Image.open('../test/图标点选/图形_{}.png'.format(j))

            probability = model.detect_image(image_1, image_2)

            #相似度低的就直接排除了
            if probability[0] >0.5:
                print('背景图__切割后图片_{}.png'.format(i),'和','图形_{}.png'.format(j),'相似度为：',probability)

        # print(image_1_name,'和',image_2_name,'相似度最高')
