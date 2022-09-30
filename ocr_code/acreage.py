'''
面积点选验证码

'''



import copy

from PIL import Image
import numpy as np
import cv2


class Acreage():

    def __init__(self):
        self.name = './data/面积点选/1.png'


    #寻找最大连通域算法
    def select_max_region(self,mask):
        nums, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        background = 0
        for row in range(stats.shape[0]):
            if stats[row, :][0] == 0 and stats[row, :][1] == 0:
                background = row
        stats_no_bg = np.delete(stats, background, axis=0)
        max_idx = stats_no_bg[:, 4].argmax()
        max_region = np.where(labels==max_idx+1, 1, 0)

        return max_region

    #加粗算法
    def expansion_px(self,img_arrays,n,h,w):
        for i in range(-n,n+1):
            for j in range(-n,n+1):
                try:
                    img_arrays[h + i, w+j] = (255, 255, 255)
                except:
                    continue
        return img_arrays

    #二值化
    def binaryzation(self):

        img = cv2.imdecode(np.fromfile(self.name, dtype=np.uint8), 1)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        #以下参数7和15和0.04都可以改，修改后识别效果会发生改变，目前的参数不确定是否为最佳参数！
        dst = cv2.cornerHarris(gray, 7, 15, 0.04)
        img[dst > 0.04 * dst.max()] = [255, 255, 255]
        img[dst < 0.04 * dst.max()] = [0, 0, 0]


        cv2.imencode('.png', img)[1].tofile(self.name)



    #灰度
    def gray(self,):
        image = cv2.imdecode(np.fromfile(self.name, dtype=np.uint8), 1)
        cv2.imencode('.png', image)[1].tofile('./data/面积点选/drow_rectangle.png')

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imencode('.png', gray)[1].tofile(self.name)

        #加粗分割线,使其形成多个独立的区域
    def zoning(self,n=7):

        '''

        :param n: 加粗像素大小，默认为7，小了没效果，大了太慢
        :return:
        '''

        img = Image.open( self.name)

        img_array = np.array(img)  # 把图像转成数组格式img = np.asarray(image)
        img_arrays = copy.deepcopy(img_array)
        shape = img_array.shape

        height = shape[0]
        width = shape[1]
        dst = np.zeros((height, width, 3))
        for h in range(0, height):
            for w in range(0, width):
                (b, g, r) = img_array[h, w]

                if b == 255 and g ==255  and r==255:

                    for i in range(1,n+1):
                        img_arrays = self.expansion_px(img_arrays,i,h,w)




                dst[h, w] = img_arrays[h, w]
        img2 = Image.fromarray(np.uint8(dst))
        img2.save( self.name, "png")

    #选出最大面积
    def  max_area(self,):
        img = cv2.imdecode(np.fromfile(self.name, dtype=np.uint8), 1)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
        mask = self.select_max_region(mask)
        cv2.imencode('.png', mask * 255)[1].tofile(self.name)




    #给出点击坐标
    def x_y_xp(self,):
        import cv2

        img = cv2.imdecode(np.fromfile(self.name, dtype=np.uint8), 1)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for i in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            rect = cv2.minAreaRect(contours[i])
            cv2.circle(img, (int(rect[0][0]), int(rect[0][1])), 2, (0, 0, 255), 2)
            return int(rect[0][0]), int(rect[0][1])


    #反转颜色
    def negative(self,):
        img = cv2.imdecode(np.fromfile(self.name, dtype=np.uint8), 1)

        img_shape = img.shape  # 图像大小(565, 650, 3)
        # 彩色图像转换为灰度图像（3通道变为1通道）
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 最大图像灰度值减去原图像，即可得到反转的图像
        dst = 255 - gray

        cv2.imencode('.png', dst )[1].tofile(self.name)



    #标识点击位置
    def drow_rectangle(self,coordinate):
        img = cv2.imdecode(np.fromfile('./data/面积点选/drow_rectangle.png', dtype=np.uint8), 1)
        # 画框
        result = cv2.rectangle(img, [coordinate[0], coordinate[0]+15],[coordinate[1], coordinate[1]+15], (0, 0, 255), 2)
        # cv2.imwrite("./data/滑动拼图/drow_rectangle.jpg", result)  # 返回圈中矩形的图片
        cv2.imencode('.png', result)[1].tofile("data/面积点选/drow_rectangle.png")

        print("返回坐标矩形成功")

    def mains(self,):
        self.gray()
        self.binaryzation()
        self.zoning()
        self.max_area()

        self.negative()
        self.max_area()
        data = self.x_y_xp()
        self.drow_rectangle(data)
        print(data)
        return data


if __name__ == '__main__':
    a = Acreage()
    data = a.mains()
    print(data)
