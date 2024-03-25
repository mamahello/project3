from PIL import Image, ImageOps
import numpy as np
import os, sys
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定

def dealimg(img_path):
    # 加载图像
    img = Image.open(img_path)

    # 如果图像是彩色的，转换为灰度图像
    if img.mode != 'L':
        img = img.convert('L')

    # 使用阈值进行二值化处理
    threshold = 127
    img = ImageOps.invert(img)  # 反转图像，使得白色背景变为黑色，数字为白色
    img = img.point(lambda p: 255 if p > threshold else 0, '1')  # 设置阈值进行二值化

    # 调整图像尺寸为28x28像素
    img = img.resize((28, 28), Image.LANCZOS)

    # 保存处理后的图像
    #output_path = 'C:\mypythonprogram\processed_image1.png'
    output_path = 'C:\\Users\\marenkun\\Desktop\\picture2\\processed_image8.png'
    img.save(output_path)

    # 如果需要，将图像数据转换为NumPy数组
    img_array = np.array(img)
    #img_array = np.array(img).reshape(784)
    return img_array


