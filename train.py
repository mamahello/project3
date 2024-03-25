import torch

from trainer import train_ch6, try_gpu
from dataset_mnist import load_data_mnist
#from Net.ResNet_18 import net
from Net.TwoLayerNet import net

lr, num_epochs, batch_size = (0.1, 200, 100)   # epoch数目 遍历了17次数据
train_iter, test_iter = load_data_mnist(batch_size)
#调整图像大小可能会导致图像内容的一些失真，因为插值算法需要在新的像素位置
#上估计像素值。然而，对于手写数字识别这样的任务，这种失真通常是可以接受的，
#并且可能不会对模型的性能产生显著影响。
train_ch6(net, train_iter, test_iter, num_epochs, lr, try_gpu())



"""

torch.save(net.state_dict(), "parameters.pth")

# 如果你还想保存模型的架构信息，可以这样做：  
torch.save(model, 'model.pth')

# 加载模型参数  
model = MyModel()  # 需要先实例化相同的模型结构  
model.load_state_dict(torch.load('model_parameters.pth'))

# 如果你保存了整个模型（包括结构），你可以这样加载：  
# model = torch.load('model.pth')  
# 此时，你不需要再实例化MyModel，因为整个模型的结构和参数都已经被加载了。

"""