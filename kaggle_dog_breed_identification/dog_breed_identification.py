import os
import torch
import torchvision
from torchvision.models import resnet34, ResNet34_Weights
from torch import nn
from d2l import torch as d2l
import gpu

data_dir = os.path.join('C:\project3\data\dog-breed-identification')

# 整理数据 并拆分训练集
def reorg_dog_data(data_dir, valid_ratio):
    labels = d2l.read_csv_labels(os.path.join(data_dir, 'labels.csv'))
    d2l.reorg_train_valid(data_dir, labels, valid_ratio)
    d2l.reorg_test(data_dir)

# 如果使用完整数据集 则批大小改为128
batch_size = 128

# 训练集比例
valid_ratio = 0.1
reorg_dog_data(data_dir, valid_ratio)   # 读取训练数据标签、拆分验证集并整理训练集。

# 数据增广
transform_train = torchvision.transforms.Compose([
    # 随机裁剪图像，所得图像为原始面积的0.08～1之间，高宽比在3/4和4/3之间。
    # 然后，缩放图像以创建224x224的新图像
    torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                             ratio=(3.0/4.0, 4.0/3.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    # 随机更改亮度，对比度和饱和度
    torchvision.transforms.ColorJitter(brightness=0.4,
                                       contrast=0.4,
                                       saturation=0.4),
    # 添加随机噪声
    torchvision.transforms.ToTensor(),
    # 标准化图像的每个通道
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])

#测试时，只使用确定的图像预处理工作
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    # 从图像中心裁切224x224大小的图片
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    # 标准化图像的每个通道
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])


# 读取数据集
#当使用 torchvision.datasets.ImageFolder 加载数据后，数据并没有直接加载到内存中的某个特定位置，
# 而是被封装在了一个 Dataset 对象中。这个 Dataset 对象是一个可迭代对象，它包含了数据集中所有样本的信息，如图像的路径、标签等。
train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_train) for folder in ['train', 'train_valid']]

valid_ds, test_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_test) for folder in ['valid', 'test']]


# 创建数据加载实例器
# 虽然可以直接迭代 dataset 来获取数据，但使用 torch.utils.data.DataLoader 可以提供更加高效、灵活和易用的数据加载方式。
train_iter, train_valid_iter = [torch.utils.data.DataLoader(
    dataset, batch_size, shuffle=True, drop_last=True)
    for dataset in (train_ds, train_valid_ds)]

valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False,
                                         drop_last=True)

test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False,
                                        drop_last=False)

# 微调预训练模型
# 深度学习框架的高级API提供了在ImageNet数据集上预训练的各种模型。此数据集是imagenet的子集
# 预训练的ResNet-34模型，我们只需重复使用此模型的输出层（即提取的特征）的输入。用一个可以训练的小型自定义输出网络替换原始输出层，例如堆叠两个完全连接的图层。
# 不重新训练用于特征提取的预训练模型，这节省了梯度下降的时间和内存空间。
# 用一个可以训练的小型自定义输出网络替换原始输出层，例如堆叠两个完全连接的图层。
def get_net(devices):
    finetune_net = nn.Sequential()
    finetune_net.features = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    # 定义一个新的输出网络，共有120个输出类别
    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 120))
    # 最后还差softmax层
    # 将模型参数分配给用于计算的CPU或GPU
    finetune_net = finetune_net.to(devices[0])
    # 冻结参数
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net

devices = gpu.try_all_gpus()
finetune_net = get_net(devices)
""""
在计算损失之前，我们首先获取预训练模型的输出层的输入，即提取的特征。
然后我们使用此特征作为我们小型自定义输出网络的输入来计算损失。
"""
loss = nn.CrossEntropyLoss(reduction='none')

def evaluate_loss(data_iter, net, devices):
    l_sum, n = 0.0, 0
    for features, labels in data_iter:
        features, labels = features.to(devices[0]), labels.to(devices[0])
        outputs = net(features)
        l = loss(outputs, labels)
        l_sum += l.sum()
        n += labels.numel()
    return (l_sum / n).to('cpu')


# 定义训练函数
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    # 只训练小型自定义输出网络
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    # 优化器
    trainer = torch.optim.SGD((param for param in net.parameters()
                               if param.requires_grad), lr=lr,momentum=0.9, weight_decay=wd)

    # 学习率调度器（实现学习率衰减）
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)

    num_batches = len(train_iter)

    #  图例 legend
    legend = ['train loss']
    if valid_iter is not None:
        legend.append('valid loss')

    for epoch in range(num_epochs):
        # 初始化一个累加器
        metric = d2l.Accumulator(2)
        for i, (features, labels) in enumerate(train_iter):
            # 将数据和标签转移到指定的设备上（如GPU）
            features, labels = features.to(devices[0]), labels.to(devices[0])

            # 清零梯度
            trainer.zero_grad()

            output = net(features)
            l = loss(output, labels).sum()
            l.backward()
            trainer.step()  # 优化器更新参数
            metric.add(l, labels.shape[0])   # 累加损失函数和训练数据总数
            """ 
            每隔5个epoch更新一次画图器
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1], None))
            """
        measures = f'train loss {metric[0] / metric[1]:.3f}'

        # 如果验证集非空 计算验证集损失函数
        if valid_iter is not None:
            valid_loss = evaluate_loss(valid_iter, net, devices)

        scheduler.step()     #学习率衰减

    if valid_iter is not None:
        measures += f', valid loss {valid_loss:.3f}'       # 通过拼接字符串的方式 在后面加上验证损失
    print(measures)


# 训练和验证模型(调节超参数部分 先跳过)
num_epochs, lr, wd = 10, 1e-4, 1e-4
lr_period, lr_decay = 2, 0.9
net = finetune_net

#train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
#      lr_decay)


# 对测试集分类并在kaggle提交结果
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
torch.save(net.state_dict(), 'parameters_resnet_34_dog.pth')

# 创建列表记录预测结果
preds = []

# 遍历测试集
for data, label in test_iter:
    output = torch.nn.functional.softmax(net(data.to(devices[0])), dim=1)       # softmax层

    """
    在PyTorch中，detach()是一个常用于张量（tensor）的方法，它的主要作用是创建一个新的张量，这个
    新张量与原始张量有相同的数据但不需要计算梯度。这常常用于切断计算图的一部分，
    防止梯度通过这部分反向传播。
    具体来说，当你有一个计算图，并且你只想对图的一部分进行梯度计算时，detach()
    非常有用。通过将某些张量与计算图“分离”，你可以防止梯度流过那些张量。
    1.output.cpu()：将输出从GPU（如果有的话）转移到CPU上。
    2.output.detach()：使用detach()方法创建一个新的张量，该张量与计算图分离，因此不需要计算梯度。
    这通常用于减少内存使用，尤其是在不需要反向传播的情况下。
    3.output.numpy()：将张量转换为NumPy数组，这样它就可以被添加到preds列表中。
    SCV文件不能直接写入张量，需要先转换成numpy数组，
    """
    preds.extend(output.cpu().detach().numpy())

# sorted() 是 Python 中的一个内置函数，用于对可迭代对象（如列表、元组、字典等）进行排序。它返回一个新的排序后的列表，而原始列表不会被改变
# os.listdir 是Python的 os 模块中的一个函数，用于列出指定目录中的所有文件和子目录的名称
ids = sorted(os.listdir(
    os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')))
with open('C:\project3\kaggle_dog_breed_identification\submission.csv', 'w') as f:
    """
    在Python中，str.join(iterable) 方法用于将一个可迭代对象（如列表或元组）中的字符串元素连接成一个单独的字符串，
    其中 str 是用作分隔符的字符串。
    这里的操作是将 train_valid_ds.classes 这个可迭代对象（通常是一个包含字符串的列表）中的所有元素用逗号 , 连接起来。  
    """
    f.write('id,' + ','.join(train_valid_ds.classes) + '\n')
    for i, output in zip(ids, preds):
        f.write(i.split('.')[0] + ',' + ','.join(
            [str(num) for num in output]) + '\n')