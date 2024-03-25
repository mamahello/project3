import collections
import math
import os
import shutil
import pandas as pd
import torch
import torchvision
from torch import nn
from d2l import torch as d2l

                                                    # 下载数据
d2l.DATA_HUB['cifar10_tiny'] = (d2l.DATA_URL + 'kaggle_cifar10_tiny.zip',
                                '2068874e4b9a9f0fb07ebe0ad2b29754449ccacd')
#为了便于入门，[**我们提供包含前1000个训练图像和5个随机测试图像的数据集的小规模样本**]。
#要使用Kaggle竞赛的完整数据集，需要将以下`demo`变量设置为`False`。
# 如果使用完整的Kaggle竞赛的数据集，设置demo为False
demo = False

if demo:
    data_dir = d2l.download_extract('cifar10_tiny')
else:
    data_dir = 'C:/project3_data/cifar-10'



                                                   # 整理数据
def read_csv_labels(fname):
    """读取fname来给标签字典返回一个文件名"""
    with open(fname, 'r') as f:
        # 跳过文件头行(列名)
        lines = f.readlines()[1:]
        # readlines() 方法读取文件的所有行，并返回一个列表。[1:]对列表进行切片跳过第一行（通常是列名）。
    tokens = [l.rstrip().split(',') for l in lines]
    """
    这行代码对每一行执行两个操作：首先使用 rstrip() 去除行尾的空白字符（如换行符），然后使用 split(',') 将行按照逗号分割成一个列表。
    最终，tokens 是一个列表的列表，其中每个内部列表都是一个由逗号分隔的字符串组成的列表。
    """
    return dict(((name, label) for name, label in tokens))
    """
    这行代码使用字典推导式将 tokens 列表转换为一个字典。假设CSV文件的格式是“文件名,标签”，
    那么每个内部列表的第一个元素（name）将作为键，第二个元素（label）将作为值。
    """

labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
print('# 训练样本 :', len(labels))
print('# 类别 :', len(set(labels.values())))
# set(labels.values()) 将所有的值转换成一个集合，自动去除了重复的元素。因此，这个集合的长度就是不同类别的数量。




                                                  # 拆分验证集
def copyfile(filename, target_dir):
    """将文件复制到目标目录"""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)

def reorg_train_valid(data_dir, labels, valid_ratio):
    """将验证集从原始的训练集中拆分出来"""
    # 训练数据集中样本最少的类别中的样本数
    n = collections.Counter(labels.values()).most_common()[-1][1]
    # 使用 collections.Counter 来统计 labels 字典中所有值的出现次数，most_common()[-1][1] 则取得出现次数最少的次数。

    # 验证集中每个类别的样本数
    # 计算每个类别在验证集中应有的样本数。这里先计算 n 与 valid_ratio 的乘积，然后向下取整，最后确保至少为1。
    n_valid_per_label = max(1, math.floor(n * valid_ratio))

    label_count = {}
    # 初始化一个空字典，用于记录每个类别在验证集中已经有多少样本

    for train_file in os.listdir(os.path.join(data_dir, 'train')):  # 遍历 data_dir 下 train 子目录中的所有文件

        label = labels[train_file.split('.')[0]]  # 获取当前文件的标签。这里假设文件名（不包括扩展名）是 labels 字典的键。

        fname = os.path.join(data_dir, 'train', train_file)  # 构造当前文件的完整路径。

        copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                     'train_valid', label))  # 将当前文件复制到新的训练目录（train_valid/label）中。

        # 判断当前类别的验证集样本数是否达到 n_valid_per_label。
        # 如果当前类别的验证集样本数未达到 n_valid_per_label，则将文件也复制到验证集目录（valid/label）。
        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'train', label))
    return n_valid_per_label




                                            # 整理测试集 方便读取
def reorg_test(data_dir):
    """在预测期间整理测试集，以方便读取"""
    # 遍历 data_dir 下 test 子目录中的所有文件。
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        #       将测试集中的每个文件复制到新的测试目录（train_valid_test/test/unknown）中。
        # 这里假设测试集中的样本没有明确的标签，因此都放在 unknown 子目录下。
        copyfile(os.path.join(data_dir, 'test', test_file),
                 os.path.join(data_dir, 'train_valid_test', 'test',
                              'unknown'))


                     # 定义一个函数 调用前面定义的`read_csv_labels`、`reorg_train_valid`和`reorg_test`
def reorg_cifar10_data(data_dir, valid_ratio):
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)



                          # 设置批大小   若用完整的数据集 则用128
batch_size = 32 if demo else 128
valid_ratio = 0.1
reorg_cifar10_data(data_dir, valid_ratio)


                            # 设置图像增广
# 训练时的图像处理
transform_train = torchvision.transforms.Compose([
    # 在高度和宽度上将图像放大到40像素的正方形
    torchvision.transforms.Resize(40),
    # 随机裁剪出一个高度和宽度均为40像素的正方形图像，
    # 生成一个面积为原始图像面积0.64～1倍的小正方形，
    # 然后将其缩放为高度和宽度均为32像素的正方形
    torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
                                                   ratio=(1.0, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    # 标准化图像的每个通道
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])

# 测试时的图像处理  测试时只执行标准化
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])


"""
torchvision.datasets.ImageFolder 和 torch.utils.data.DataLoader 在 PyTorch 中是两个不同的工具，
经常一起使用以构建高效的数据加载和预处理管道。

torchvision.datasets.ImageFolder
torchvision.datasets.ImageFolder 是一个用于从文件夹中加载图像数据的类。，用于从文件夹结构中加载图像数据集。它假设每个子文件夹包含同一类别的
图像。它会根据文件夹的名称自动为图像打上标签。
例如，如果 'train' 文件夹下有一个名为 'dog' 的子文件夹，那么该子文件夹中的所有图像都将被标记为 'dog'。

列表推导式
[... for folder in ['train', 'train_valid']] 是一个列表推导式，它遍历列表 ['train', 'train_valid'] 中的每个元素（folder）。
对于每个 folder，它都会创建一个 ImageFolder 数据集对象，并将这些对象收集到一个列表中。由于列表推导式只包含一个元素，所以实际上生成的
是一个包含两个数据集的元组。然后，这个元组被解包成 train_ds 和 train_valid_ds 这两个变量。

torch.utils.data.DataLoader
torch.utils.data.DataLoader 是一个用于加载数据集的迭代器。它提供了一个可迭代的对象，
使得你可以在一个训练循环中批量地获取数据。
"""

                            # 加载数据
# torchvision.datasets.ImageFolder  它用于从目录结构中加载图像数据
train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_train) for folder in ['train', 'train_valid']]

valid_ds, test_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_test) for folder in ['valid', 'test']]


                            # 用于加载数据的迭代器
train_iter, train_valid_iter = [torch.utils.data.DataLoader(
    dataset, batch_size, shuffle=True, drop_last=True)
    for dataset in (train_ds, train_valid_ds)]

valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False,
                                         drop_last=True)

test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False,
                                        drop_last=False)


                                #定义模型 损失函数
def get_net():
    num_classes = 10
    net = d2l.resnet18(num_classes, 3)
    return net

loss = nn.CrossEntropyLoss(reduction="none")
# reduction设置为"none"时，损失函数将为每个样本单独计算损失，并
# 返回一个形状与输入标签相同的张量，而不是返回所有样本损失的平均值或总和



                            # 定义训练函数
# 定义一个函数train，接受多个参数，包括神经网络模型net、训练数据迭代器train_iter、验证数据迭代器valid_iter、
# 训练轮数num_epochs、学习率lr、权重衰减wd、设备列表devices、学习率调整周期lr_period和学习率衰减率lr_decay。
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    # 优化器
    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                              weight_decay=wd)
    # 学习率调度器 用于在每个lr_period周期后，将学习率乘以lr_decay（学习率衰减）
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)

    num_batches, timer = len(train_iter), d2l.Timer()

    # 初始化图例列表，包括训练损失和训练准确率
    legend = ['train loss', 'train acc']
    if valid_iter is not None:
        legend.append('valid acc')  # 如果提供了验证迭代器，则在图例列表中添加验证准确率

    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)

    # 使用DataParallel将模型并行化到多个GPU设备上
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])

    for epoch in range(num_epochs):
        net.train()     # 训练模式

        #初始化累加器
        #累加三个变量 损失函数 预测正确数 样本数量
        metric = d2l.Accumulator(3)

        # 遍历训练数据迭代器
        for i, (features, labels) in enumerate(train_iter):
            # 开始计时
            timer.start()

            l, acc = d2l.train_batch_ch13(net, features, labels,
                                          loss, trainer, devices)

            # 累加器累加
            metric.add(l, acc, labels.shape[0])

            timer.stop()

            # 每5个epoch 或最后一个epoch 在动画器中加当前轮数、训练损失和训练准确率
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[2],
                              None))

        # 如果验证集非空 如果提供了验证迭代器，则进行验证。
        if valid_iter is not None:
            # 用验证数据评估模型准确率  并在动画器中添加验证准确率
            valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)
            animator.add(epoch + 1, (None, None, valid_acc))    # 画图器更新验证集的准确率

        #更新学习率
        scheduler.step()

        # 总并打印一个训练周期内的训练损失函数、训练精度和验证结果（如果有的话），以及整个训练过程的平均吞吐量。
        # 这对于监控模型的训练进度和性能非常有用。
        measures = (f'train loss {metric[0] / metric[2]:.3f}, '
                    f'train acc {metric[1] / metric[2]:.3f}')
        if valid_iter is not None:
            measures += f', valid acc {valid_acc:.3f}'
        print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f}'
              f' examples/sec on {str(devices)}')



                                # 训练和验证模型
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 50, 2e-4, 5e-4
lr_period, lr_decay = 4, 0.9

                                # 在kaggle上对测试集进行分类并提交结果
net = get_net()     # 初始化模型
preds = []          # 初始化列表记录测试数据的预测结果
net = net.to(devices[0])  # 把模型移动到GPU上

# 训练模型
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
torch.save(net.state_dict(), 'parameters_resnet_18_cifar10.pth')
# 遍历测试数据
for X, _ in test_iter:
    y_hat = net(X.to(devices[0]))
    preds.extend(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy())
    # append()添加一个元素  extend（）方法在列表的末尾一次性添加多个元素（这些元素通常来自另一个列表或可迭代对象）
    """
    由于模型的输出通常是每个类别的概率，所以通过argmax(dim=1)获取概率最大的类别的索引，即预测结果。
    接着，将预测结果转换为整数类型（torch.int32），并从GPU移至CPU，然后转换为NumPy数组，最后添加到preds列表中。
    """

sorted_ids = list(range(1, len(test_ds) + 1))   # 测试集每个样本的序号列表

sorted_ids.sort(key=lambda x: str(x))           # 暂无意义？

df = pd.DataFrame({'id': sorted_ids, 'label': preds})
"""
pandas 库创建了一个名为 df 的DataFrame。DataFrame是一个二维的、大小可变的、有潜在异构类型列的表格型数据结构。
这里，DataFrame有两列：'id' 和 'label'。'id' 列包含 sorted_ids 列表中的ID，而 'label' 列包含 preds 列表中
的模型预测结果。
"""
# 这行代码将 df DataFrame中 'label' 列的预测结果（通常是类别的索引）转换为实际的类别标签。
# lambda 函数接收一个参数 x（即 'label' 列中的一个元素），然后使用这个参数作为索引从 train_valid_ds.classes
# 中获取对应的类别标签。这样，'label' 列中的每个数字索引就被替换为了实际的类别名称或标签。
df['label'] = df['label'].apply(lambda x: train_valid_ds.classes[x])

df.to_csv('submission.csv', index=False)        # 不包含DataFrame的索引（index=False）。

