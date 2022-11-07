'''
调用案例

'''

import requests
import base64
import cv2
import numpy as np
from PIL import Image


#将RGBA改为RGB（有些图片只有透明度没有颜色，所以需要处理下）
def qweada(path):
        try:
            # img2 = cv2.imread(path , cv2.IMREAD_UNCHANGED)
            img2 = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)

            b, g, r, a = cv2.split(img2)
            aa = a[:, :, np.newaxis]
            new = np.concatenate((aa, aa, aa), axis=2)
            im = Image.fromarray(new)
            im.save(path)
        except:
            return



#图标点选
def tu_biao():
    img1 = base64.b64encode(    open('图标点选/1.jpg','rb').read()).decode()

    qweada('图标点选/2.png')
    with open('图标点选/2.png','rb')as f:
        img2_1 = f.read()
    img2_1 = base64.b64encode(img2_1).decode()


    qweada('图标点选/3.png')
    with open('图标点选/3.png','rb')as f:
        img2_2 = f.read()
    img2_2 = base64.b64encode(img2_2).decode()


    qweada('图标点选/4.png')
    with open('图标点选/4.png','rb')as f:
        img2_3 = f.read()
    img2_3 = base64.b64encode(img2_3).decode()


    data ={
        'img1':img1,
        'img2':[
            img2_1,
            img2_2,
            img2_3,
        ],
        'show': True  # 此参数可以不携带  为True时，会旋转图片

    }
    resonse = requests.post('http://119.3.219.181:8000/图标点选/',json=data).text
    print(resonse)

#滑块拼图·
def hua_kuai():

    with open('滑块拼图/123.png','rb')as f:
        img = f.read()
        img = base64.b64encode(img).decode()

    data = {
        'img':img,
        'show':True        #此参数可以不携带  为True时，会在图片上画一个红框

    }
    response = requests.post('http://119.3.219.181:8000/滑块拼图/',data=data).json()
    print(response)

#面积点选
def mian_ji():

    with open('面积点选/1.png','rb')as f:
        img = f.read()
        img = base64.b64encode(img).decode()


    data = {
        'img':img,
        'show':True        #此参数可以不携带  为True时，会在图片上画一个红框

    }
    response = requests.post('http://119.3.219.181:8000/面积点选/',data=data).json()
    print(response)


#角度旋转
def xuanz_zhuan():
    with open('旋转/1.png','rb')as f:
        img = f.read()
        img = base64.b64encode(img).decode()
    data = {
        'img':img,
        'show':'True'
    }
    resource = requests.post('http://119.3.219.181:8000/旋转图片/',data).json()
    print(resource)


if __name__ == '__main__':
    '''
    调用案例示范~ 
    
    '''
    tu_biao()
    # hua_kuai()
    # mian_ji()
    # xuanz_zhuan()


    #  pscp  -r D:\工作\simple_ocr\models  root@119.3.219.181:/opt/simple_ocr