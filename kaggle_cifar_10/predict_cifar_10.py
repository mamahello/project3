import d2l.torch as d2l
import torch
import torchvision
from PIL import Image

def get_net():
    num_classes = 10
    net = d2l.resnet18(num_classes, 3)
    return net

work_net = get_net()
work_net.eval()
work_net.load_state_dict(torch.load('parameters_resnet_18_cifar10.pth'))

devices = d2l.try_gpu()
work_net = work_net.to(devices)  # 把模型移动到GPU上

# 图像处理
# 测试时的图像处理  测试时只执行标准化
# 定义与CIFAR-10相同的预处理步骤
# CIFAR-10通常使用的预处理包括：归一化到[0, 1]，然后减去均值并除以标准差
# CIFAR-10的均值和标准差为：(0.4914, 0.4822, 0.4465) 和 (0.2023, 0.1994, 0.2010)
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])])

# 加载图片
image_path = 'C:\\Users\\marenkun\\Desktop\\cifar_10\\2.jpg'  # 替换为你的图片路径
image = Image.open(image_path)

# 确保图像是RGB模式(网上下的图是4通道）
image = image.convert('RGB')
# 应用预处理步骤
processed_image = transform_test(image)
processed_image = processed_image.unsqueeze(0)

processed_image = processed_image.cuda()

# 进行前向传播
with torch.no_grad():  # 如果不需要计算梯度，可以使用这个上下文管理器来节省内存
    output = work_net(processed_image)  # 获取模型的输出 一个张量

# 输出output现在包含了模型对这张图片的前向传播结果
# 首先，将输出张量从GPU转移到CPU
output_cpu = output.cpu()

predict = torch.argmax(output_cpu)

# CIFAR-10类别的中文标签映射
cifar10_chinese_labels = [
    "飞机",
    "汽车",
    "鸟",
    "猫",
    "鹿",
    "狗",
    "青蛙",
    "马",
    "船",
    "卡车"
]


chinese_labels = cifar10_chinese_labels[predict]
print("这张图片的物品是：",chinese_labels)  # 输出: ['猫', '狗', '汽车', '马']