from torchvision.models import resnet34, ResNet34_Weights

# 使用默认权重（通常是最新版本的预训练权重）
model = resnet34(weights=ResNet34_Weights.DEFAULT)

# 或者，如果你想要使用特定版本的ImageNet预训练权重
model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
# 查看网络构架
#print(resnet34)
