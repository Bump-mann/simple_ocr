# simple_ocr
一个简易的识别验证码的代码
这是一个简易验证码识别系统，目的是搭建一个验证识别服务
现在还是很简陋的，且识别率较低，且普适性较差
如果遇到验证码识别率很低，请联系我改进！

交流群：QQ群 949504676


在线测试页面：http://121.4.108.95:8000/index/   服务器很拉，请勿玩坏

目前支持的验证码类型：

1、面积点选验证码

2、滑动验证码

3、旋转验证码

4、图标点选验证码



由于模型文件较大，单独放在百度云盘了，下载后放在项目目录下即可
模型下载地址：
链接：https://pan.baidu.com/s/1rxY2x3J8wwgEsv0nBBaPPQ?pwd=cokk 
提取码：cokk

开发测试环境：
  python::3.8
  3060显卡
  
  
 您可以在项目目录下 pip install -i requirements.txt 快速安装所需库
 
 
 在test文件夹下的test.py里有调用示范代码！
 
 
 我接下来将要做的（排名不分前后）
 1、收集用户反馈，改进模型（如果有用户的话）
 2、文字点选验证码
 3、差异点击验证码
 4、英文数字混合验证码
 5、算数验证码
