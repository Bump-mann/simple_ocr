r'''
百度贴吧验证码图片


'''


import tensorflow as tf
import keras.backend as K

import cv2
import numpy as np
import os
from PIL import Image


class my_ocr():
    def __init__(self):
        # physical_devices = tf.config.list_physical_devices('GPU')
        # tf.config.experimental.set_memory_growth(physical_devices[0], True)
        self.model = tf.keras.models.load_model(os.path.abspath("models/旋转验证码.hdf5"), custom_objects={'angle_error':self.angle_error})

    def angle_difference(self,x, y):
        return 180 - abs(abs(x - y) - 180)

    #
    def angle_error(self,y_true, y_pred):
        diff = self.angle_difference(K.argmax(y_true), K.argmax(y_pred))
        return K.mean(K.cast(K.abs(diff), K.floatx()))

    def round(self,img_path, times):
        ima = Image.open(img_path).convert("RGBA")
        size = ima.size
        # 要使用圆形，所以使用刚才处理好的正方形的图片
        r2 = min(size[0], size[1])
        if size[0] != size[1]:
            ima = ima.resize((r2, r2), Image.ANTIALIAS)
        # 最后生成圆的半径
        r3 = int(r2 / 2)
        imb = Image.new('RGBA', (r3 * 2, r3 * 2), (255, 255, 255, 0))
        pima = ima.load()  # 像素的访问对象
        pimb = imb.load()
        r = float(r2 / 2)  # 圆心横坐标

        for i in range(r2):
            for j in range(r2):
                lx = abs(i - r)  # 到圆心距离的横坐标
                ly = abs(j - r)  # 到圆心距离的纵坐标
                l = (pow(lx, 2) + pow(ly, 2)) ** 0.5  # 三角函数 半径
                if l < r3:
                    pimb[i - (r - r3), j - (r - r3)] = pima[i, j]


        imb.save(times)
        return


    def gray(self):
        img = Image.open('./data/旋转角度/1.png')
        Img = img.convert('L')
        Img.save('./data/旋转角度/1.png')

    #旋转角度预测
    def identification(self):
        '''

        :param img_path: 图片地址 例如：D/1.png
        :return:
        '''

        # self.round(img_path,img_path)  #将图片变圆
        img_path = './data/旋转角度/1.png'
        self.gray()  #转为灰度图片
        img_init = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)

        img_init = cv2.resize(img_init, (224, 224))  # 将图片大小调整到224*224用于模型推理
        img = np.asarray(img_init)
        outputs = self.model.predict(img.reshape(1, 224, 224, 3))
        # print(outputs)
        result_index = int(np.argmax(outputs))
        class_names = ['0', '1', '10', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '11', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '12', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129', '13', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '14', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '15', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '16', '160', '161', '162', '163', '164', '165', '166', '167', '168', '169', '17', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '18', '180', '181', '182', '183', '184', '185', '186', '187', '188', '189', '19', '190', '191', '192', '193', '194', '195', '196', '197', '198', '199', '2', '20', '200', '201', '202', '203', '204', '205', '206', '207', '208', '209', '21', '210', '211', '212', '213', '214', '215', '216', '217', '218', '219', '22', '220', '221', '222', '223', '224', '225', '226', '227', '228', '229', '23', '230', '231', '232', '233', '234', '235', '236', '237', '238', '239', '24', '240', '241', '242', '243', '244', '245', '246', '247', '248', '249', '25', '250', '251', '252', '253', '254', '255', '256', '257', '258', '259', '26', '260', '261', '262', '263', '264', '265', '266', '267', '268', '269', '27', '270', '271', '272', '273', '274', '275', '276', '277', '278', '279', '28', '280', '281', '282', '283', '284', '285', '286', '287', '288', '289', '29', '290', '291', '292', '293', '294', '295', '296', '297', '298', '299', '3', '30', '300', '301', '302', '303', '304', '305', '306', '307', '308', '309', '31', '310', '311', '312', '313', '314', '315', '316', '317', '318', '319', '32', '320', '321', '322', '323', '324', '325', '326', '327', '328', '329', '33', '330', '331', '332', '333', '334', '335', '336', '337', '338', '339', '34', '340', '341', '342', '343', '344', '345', '346', '347', '348', '349', '35', '350', '351', '352', '353', '354', '355', '356', '357', '358', '359', '36', '37', '38', '39', '4', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '5', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '6', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '7', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '8', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '9', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99']

        result = class_names[result_index]  # 获得名称
        self.rotate_img(img_init,result)
        return result




    #旋转图片
    def rotate_img(self,img, angle):
        '''
        img   --image
        angle --rotation angle
        return--rotated img
        '''
        h, w = img.shape[:2]
        rotate_center = (w / 2, h / 2)
        # 获取旋转矩阵
        # 参数1为旋转中心点;
        # 参数2为旋转角度,正值-逆时针旋转;负值-顺时针旋转
        # 参数3为各向同性的比例因子,1.0原图，2.0变成原来的2倍，0.5变成原来的0.5倍
        M = cv2.getRotationMatrix2D(rotate_center, -float(angle), 1.0)
        # 计算图像新边界
        new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
        new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
        # 调整旋转矩阵以考虑平移
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2

        rotated_img = cv2.warpAffine(img, M, (new_w, new_h))
        cv2.imencode('.png', rotated_img)[1].tofile('./data/旋转角度/drow_rectangle.png')

        return rotated_img

