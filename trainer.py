import torch
from torch import nn
from d2l import torch as d2l

def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def evaluate_accuracy_gpu(net, data_iter, device=None):   # @save
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 检查net是否是一个PyTorch的nn.Module实例。如果是，则调用eval()方法将其设置为评估模式。
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)     # 初始化一个d2l.Accumulator对象，用于累加两个值：正确预测的数量和总预测的数量。
    with torch.no_grad():
        for X, y in data_iter:
            # 检查输入数据X是否是一个列表。如果是（这可能在处理某些特殊模型，如BERT时发生），则遍历列表并将每个元素移动
            # 到指定设备device上。如果X不是列表，则直接将其移动到指定设备。同样，标签y也被移动到指定设备。
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            # 使用模型net对输入数据X进行预测，并使用d2l.accuracy计算预测精度。
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

# @save

# 训练数据，测试数据已经是从loaddate里的小批量了
# train_iter, test_iter可迭代对象，每次迭代的元素都是一个小批量 小批量数据形式：N,C,H,W
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)"""

    # 初始化参数
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)     # net的每层都运用一次初始权重函数

    print('training on', device)
    net.to(device)  # 将网络模型及其参数移动到指定的设备上

    # 实例化优化器和损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    # 画图器
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])

    timer = d2l.Timer()             # 计时器
    num_batches = len(train_iter)   # 每个批的数据个数

    for epoch in range(num_epochs):
        # 初始化一个累加器metric，用于在训练过程中累加损失、准确率和样本数
        metric = d2l.Accumulator(3)      # metric为一个累加器，可累加三个值

        net.train()     # 网络设置为训练模式

        for i, (X, y) in enumerate(train_iter):  # 遍历训练数据 x训练数据 y训练标签

            timer.start()   # 开始计时

            optimizer.zero_grad()  # 清空优化器中的梯度，为计算新的小批量的梯度做准备。

            X, y = X.to(device), y.to(device)    # 把训练数据 训练标签移动到GPU上

            y_hat = net(X)  # 前向传播获取模型的预测值y_hat，并计算损失函数
            l = loss(y_hat, y)

            l.backward()            # 通过反向传播计算梯度

            optimizer.step()        # 优化器更新模型参数

            # 在不计算梯度的情况下，计算当前小批量的损失、准确数和样本数，并将它们累加到metric中
            # x.shape[0]为batchsize,即样本数量
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
                # l*

            timer.stop()    # 停止计时

            # 计算小批量的平均损失，训练的准确率
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]

            # 每隔一定数量的批次（这里是每5个批次,//整除运算），或者在最后一个批次，更新动画器中的损失
            # （横坐标epoch,纵坐标损失函数、训练精度、测试精度）
            # 和训练准确率曲线。None用于表示测试准确率，因为在训练过程中通常不计算测试准确率。
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))

        test_acc = evaluate_accuracy_gpu(net, test_iter)    # GPU上计算测试精度，并且更新动画器animator
        animator.add(epoch + 1, (None, None, test_acc))

    # 每个epoch打印输出 f字符串
        print(f'loss {train_l:.4f}, train acc {train_acc:.4f}, '
          f'test acc {test_acc:.4f}')

        print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')