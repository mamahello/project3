import d2l.torch as d2l
import torch
import torchvision
from PIL import Image



# 加载图片
image_path = 'C:\\Users\\marenkun\\Desktop\\cifar_10\\1.jpg'  # 替换为你的图片路径
image = Image.open(image_path)


# 应用预处理步骤
resize = torchvision.transforms.Resize((32, 32))
image = resize(image)
image.show()


