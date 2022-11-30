import cv2
import numpy as np
import json
from django.http import HttpResponse
import base64
from PIL import Image


from ocr_code import spin_img
from ocr_code import acreage
from ocr_code import hua_kuai
from ocr_code import img_similarity
from ocr_code import img_division



#如果不使用旋转验证码识别可注释掉
spin_img = spin_img.my_ocr()
hua_kuai = hua_kuai.slider()
img_qwe = img_division.img_qwe()
model = img_similarity.Siamese()


#去干扰 保留数量最多的灰度及其附近值
def  decouple(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 0)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lists = []
    for i in img:
        for j in img:
            for k in j:
                lists.append(k)

    dicts = {i: 0 for i in range(0, 256)}

    for i in lists:
        dicts[i] = dicts.get(i, 0) + 1
        # dicts.get(i,1)

    def getDictKey_1(myDict, value):
        return [k for k, v in myDict.items() if v == value]

    max_num = getDictKey_1(dicts, max(dicts.values()))[0] - 5


    for i in range(len(img)):
        for j in range(len(img[i])):

            if img[i][j] > max_num + 10:
                img[i][j] = 0
            if img[i][j] < max_num - 10:
                img[i][j] = 0
    ret, binary = cv2.threshold(img, max_num, 255, cv2.THRESH_BINARY)  # 输入灰度图，实现图像二值化

    cv2.imencode('.png', binary)[1].tofile(path)



# 旋转
def ocr(request):
    resp = {}

    if request.method == 'POST':
        img = request.POST.get('img')
        if img:
            img = base64.b64decode(img)
        else:
            resp['errorcode'] = '请携带img参数，img:base64编码后的图片二进制'
            return HttpResponse(json.dumps(resp), content_type="application/json")

        with open('./data/旋转角度/1.png', 'wb') as f:
            f.write(img)
        f.close()

        result = spin_img.identification()
        resp['detail'] = result
        resp['explain'] = '顺时针旋转角度'




        return HttpResponse(json.dumps(resp), content_type="application/json")
    else:
        resp = {'errorcode': 100, 'detail': 'get啥呢，去post'}
        return HttpResponse(json.dumps(resp), content_type="application/json")


# 面积
def acreages(request):
    resp = {}

    if request.method == 'POST':
        img = request.POST.get('img')
        if img:
            img = base64.b64decode(img)
        else:
            resp['errorcode'] = '请携带img参数，img:base64编码后的图片二进制'
            return HttpResponse(json.dumps(resp), content_type="application/json")

        with open('./data/面积点选/1.png', 'wb') as f:
            f.write(img)
        f.close()


        acreages = acreage.Acreage()
        result = acreages.mains()
        resp['detail'] = result
        resp['explain'] = '点击位置的坐标'

        return HttpResponse(json.dumps(resp), content_type="application/json")
    else:
        resp = {'errorcode': 100, 'detail': 'get啥呢，去post'}
        return HttpResponse(json.dumps(resp), content_type="application/json")


# 滑块
def slider(request):
    resp = {}

    if request.method == 'POST':
        img = request.POST.get('img')
        if img:
            img = base64.b64decode(img)
        else:
            resp['errorcode'] = '请携带img参数(带缺口的背景图片)，img:base64编码后的图片二进制'
            return HttpResponse(json.dumps(resp), content_type="application/json")

        with open('./data/滑块拼图/1.png', 'wb') as f:
            f.write(img)
        f.close()
        distence = hua_kuai.onnx_model_main('./data/滑块拼图/1.png')
        hua_kuai.drow_rectangle(distence,'./data/滑块拼图/1.png')
        distence = int(distence['leftTop'][0])

        resp['detail'] = distence
        resp['explain'] = '图片左边框到缺口左边框距离'





        return HttpResponse(json.dumps(resp), content_type="application/json")
    else:
        resp = {'errorcode': 100, 'detail': 'get啥呢，去post'}
        return HttpResponse(json.dumps(resp), content_type="application/json")


# 图标点选
def click_on_the_icon(request):
    if request.method == 'POST':

        json_dict = json.loads(request.body.decode())
        img1 = base64.b64decode(json_dict['img1'])
        with open('./data/图标点选/背景图.png','wb')as f:
                f.write(img1)

        num = 0
        for i in json_dict['img2']:
            with open('./data/图标点选/图形_{}.png'.format(num), 'wb') as f:

                f.write(base64.b64decode(i))
            num+=1
        path = './data/图标点选/背景图.png'
        coordinate_onnx = img_qwe.onnx_model_main(path)

        num = 0
        #矩形坐标列表
        lists = []
        for j in coordinate_onnx:

            lists.append(j['leftTop']+j['rightBottom'])
            image = Image.open(path)  # 读取图片
            name = path[:-4:] + '__切割后图片_' + str(num)
            img_division.cut_image(image, j['point'], name)
            num += 1

        #图形数量
        num = len( json_dict['img2'])
        #切割数量
        nums = len(coordinate_onnx)
        print(coordinate_onnx)
        resp = {}


        for i in range(nums):
            decouple('./data/图标点选/背景图__切割后图片_{}.png'.format(i))

        for i in range(num):

            image_1 = Image.open('./data/图标点选/图形_{}.png'.format(i))
            max_probability = 0
            for j in range(nums):

                image_2 = Image.open('./data/图标点选/背景图__切割后图片_{}.png'.format(j))

                probability = model.detect_image(image_1, image_2)

                # 相似度低的就直接排除了
                if probability[0] > 0:
                    print('背景图__切割后图片_{}.png'.format(i), '和', '图形_{}.png'.format(j), '相似度为：',
                          probability)
                    if probability[0] > max_probability:
                        max_probability = probability[0]
                        resp['img' + str(i)] = lists[j]


            # resp['img'+str(i)] = max_probability
            print()
        print(resp)

        num = 0
        for i in resp.values():
            if num == 0:
                img = cv2.imdecode(np.fromfile('./data/图标点选/背景图.png', dtype=np.uint8), 1)
            else:
                img = cv2.imdecode(np.fromfile('./data/图标点选/drow_rectangle.png', dtype=np.uint8), 1)

        # 画框
            result = cv2.rectangle(img,i[0:2:],i[2:4:], (0, 0, 255), 2)
            cv2.putText(result, str(num), (int((i[0]+i[2])/2),int( (i[1]+i[3])/2)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 0), 1, cv2.LINE_AA)

            cv2.imencode('.png', result)[1].tofile(r"./data/图标点选/drow_rectangle.png")

            print("返回坐标矩形成功")
            num+=1

        return HttpResponse(json.dumps(resp), content_type="application/json")


    else:

        resp = {'errorcode': 100, 'detail': 'get啥呢，去post'}
        return HttpResponse(json.dumps(resp), content_type="application/json")


