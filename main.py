'''

启动服务后运行下这个
模型第一次使用会很慢很慢（大约30秒？），所以每次启动时，都要先识别一张图片来预热

'''

import requests
import base64

with open('旋转.png','rb')as f:
    img_data  = f.read()

url = 'http://127.0.0.1:8000/ocr/'

img_data = base64.b64encode(img_data)
data = {
    'img':img_data
}


response = requests.post(url=url,data=data).content.decode('utf-8')

print(response)
