
import json
from django.http import HttpResponse
from ocr_code import my_ocr
from ocr_code import acreage
from ocr_code import hua_kuai
import base64
my_ocr = my_ocr.my_ocr()


#旋转
def ocr(request):
   resp = {}

   if request.method == 'POST':
      img = request.POST.get('img')
      if img:
         img  = base64.b64decode(img)
      else:
         resp['errorcode'] = '请携带img参数，img:base64编码后的图片二进制'
         return HttpResponse(json.dumps(resp), content_type="application/json")


      with open('1.png','wb')as f:
         f.write(img)
      f.close()

      result = my_ocr.identification('./1.png')
      resp['detail'] = result
      resp['explain'] = '顺时针旋转角度'


      return HttpResponse(json.dumps(resp), content_type="application/json")
   else:
      resp = {'errorcode': 100, 'detail': 'get啥呢，去post'}
      return HttpResponse(json.dumps(resp), content_type="application/json")

#面积
def acreages(request):
   resp = {}

   if request.method == 'POST':
      img = request.POST.get('img')
      if img:
         img  = base64.b64decode(img)
      else:
         resp['errorcode'] = '请携带img参数，img:base64编码后的图片二进制'
         return HttpResponse(json.dumps(resp), content_type="application/json")


      with open('1.png','wb')as f:
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



#滑块
def slider(request):
   resp = {}

   if request.method == 'POST':
      img = request.POST.get('img')
      if img:
         img = base64.b64decode(img)
      else:
         resp['errorcode'] = '请携带img参数(带缺口的背景图片)，img:base64编码后的图片二进制'
         return HttpResponse(json.dumps(resp), content_type="application/json")

      with open('1.png', 'wb')as f:
         f.write(img)
      f.close()
      distence = hua_kuai.onnx_model_main('1.png')
      distence = int(distence['leftTop'][0])


      resp['detail'] = distence
      resp['explain'] = '图片左边框到缺口左边框距离'

      return HttpResponse(json.dumps(resp), content_type="application/json")
   else:
      resp = {'errorcode': 100, 'detail': 'get啥呢，去post'}
      return HttpResponse(json.dumps(resp), content_type="application/json")

