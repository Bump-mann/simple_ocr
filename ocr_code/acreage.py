'''
面积点选验证码

'''


import copy

from PIL import Image
import numpy as np
import cv2



class Acreage():




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
    def binaryzation(self,n=155):
        input_img_file= '1.png'
        gray = cv2.imread(input_img_file)

        # 固定阈值设置
        ret, binary = cv2.threshold(gray, n, 255, cv2.THRESH_BINARY)
        cv2.imwrite('1.png',binary)


    #灰度
    def gray(self,):
        image = cv2.imread("1.png")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('1.png', gray)


    #加粗分割线,使其形成多个独立的区域
    def zoning(self,n=7):

        '''

        :param n: 加粗像素大小，默认为7，小了没效果，大了太慢
        :return:
        '''

        img = Image.open("1.png")

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
        img2.save("1.png", "png")

    #选出最大面积
    def  max_area(self,):
        img = cv2.imread('1.png')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
        mask = self.select_max_region(mask)

        cv2.imwrite('1.png', mask * 255)


    #给出点击坐标
    def x_y_xp(self,):
        import cv2

        img = cv2.imread("1.png")
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
        img = cv2.imread('1.png', 1)
        cv2.imshow('img', img)
        img_shape = img.shape  # 图像大小(565, 650, 3)
        # 彩色图像转换为灰度图像（3通道变为1通道）
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 最大图像灰度值减去原图像，即可得到反转的图像
        dst = 255 - gray

        cv2.imwrite('1.png', dst)


    #去噪点
    def degrains(self,):

        def clamp(pv):
            if pv > 255:
                return 255
            if pv < 0:
                return 0
            else:
                return pv

        def gaussian_noise(image):  # 加高斯噪声
            h, w, c = image.shape
            for row in range(h):
                for col in range(w):
                    s = np.random.normal(0, 20, 3)
                    b = image[row, col, 0]  # blue
                    g = image[row, col, 1]  # green
                    r = image[row, col, 2]  # red
                    image[row, col, 0] = clamp(b + s[0])
                    image[row, col, 1] = clamp(g + s[1])
                    image[row, col, 2] = clamp(r + s[2])

        src = cv2.imread('1.png')


        gaussian_noise(src)
        dst = cv2.GaussianBlur(src, (15, 15), 0)  # 高斯模糊
        cv2.imwrite('1.png', dst)


    def degrain(self):

        def salt(img, n):
            for k in range(n):
                i = int(np.random.random() * img.shape[1])
                j = int(np.random.random() * img.shape[0])
                if img.ndim == 2:
                    img[j, i] = 255
                elif img.ndim == 3:
                    img[j, i, 0] = 255
                    img[j, i, 1] = 255
                    img[j, i, 2] = 255
                return img

        img = cv2.imread("./1.png", cv2.IMREAD_GRAYSCALE)
        result = salt(img, 500)
        median = cv2.medianBlur(result, 5)
        cv2.imwrite('1.png', median)

    def mains(self,):
        self.gray()
        self.binaryzation()
        self.zoning()
        self.degrain()
        self.max_area()

        self.negative()
        self.max_area()
        data = self.x_y_xp()
        return data
        # return  1

if __name__ == '__main__':
    a = Acreage()
    data = a.mains()
    print(data)