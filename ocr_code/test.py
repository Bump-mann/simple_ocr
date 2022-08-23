with open('旋转.png', 'wb')as f:
    f.write(img)
f.close()

result = my_ocr.identification('./1.png')