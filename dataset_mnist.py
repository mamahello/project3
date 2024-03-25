import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()
# 不能删不然报错，在你的代码中，DataLoader在内部使用了multiprocessing来
# 并行加载数据，所以你需要确保你的主模块代码在if __name__ == '__main__':语句块内。

def get_dataloader_workers():  # @save
    """使用4个进程来读取数据"""
    return 4

def load_data_mnist(batch_size, resize=None):  # @save      #参数 1.批大小 2.是否改变数据大小
    """下载MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.MNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.MNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))


"""
train_iter, test_iter = load_data_mnist(256, resize=None)
#其中train_iter,test_iter都是一个可迭代对象 迭代方法如下:，x数据y标签

for X, y in train_iter:
    print(X.shape, y.shape, y.dtype)
    break
"""