"""ocr_api URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path,re_path
from ocr.views import *
from test_page.views import *

urlpatterns = [
    # path('admin/', admin.site.urls),
    path('旋转图片/',ocr,name='旋转图片'),
    path('面积点选/',acreages,name='面积点选'),
    path('滑块拼图/',slider,name='滑块拼图'),
    path('图标点选/',click_on_the_icon,name='图标点选'),
    re_path('index/.*?',index,name='index'),
    path('测试/',test,name='测试')
]
