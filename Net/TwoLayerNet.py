from torch import nn

net = nn.Sequential(nn.Conv2d(1, 30, 5, 1), nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Flatten(),  # 展平操作
                    nn.Linear(30*12*12,100), nn.ReLU(),
                    nn.Linear(100, 10), nn.Softmax(dim=0)
                    )
"""
                    实现为一个类
import torch  
import torch.nn as nn  
import torch.nn.utils.rnn as rnn_utils  
  
class CNN(nn.Module):  
    def __init__(self):  
        super(CNN, self).__init__()  
        self.conv1 = nn.Conv2d(1, 30, 5, 1)  
        self.relu1 = nn.ReLU()  
        self.pool1 = nn.MaxPool2d(2, 2)  
        self.conv2 = nn.Conv2d(30, 50, 5, 1)  
        self.relu2 = nn.ReLU()  
        self.pool2 = nn.MaxPool2d(2, 2)  
        # 全连接层的输入特征数将由特征图的大小和通道数决定  
        self.fc = nn.Linear(0, 10)  # 初始时不知道输入特征数，所以设置为0  
  
    def forward(self, x):  
        x = self.conv1(x)  
        x = self.relu1(x)  
        x = self.pool1(x)  
        x = self.conv2(x)  
        x = self.relu2(x)  
        x = self.pool2(x)  
        # 展平特征图  
        x = x.view(x.size(0), -1)  # -1 表示让PyTorch自动计算该维度的大小  
        # 设置全连接层的输入特征数  
        self.fc.in_features = x.size(1)  
        x = self.fc(x)  
        return x  
  
# 创建网络实例  
net = CNN()  
  
# 假设我们有一个输入张量  
input_tensor = torch.randn(1, 1, 28, 28)  # 大小为 [batch_size, channels, height, width]  
  
# 前向传播，使网络计算特征图的尺寸  
output = net(input_tensor)  
  
# 此时，全连接层的输入特征数已经被自动设置  
print(net.fc.in_features)
"""