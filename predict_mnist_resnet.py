from Net.ResNet_18 import net
import torch
from deal_img import dealimg
from torchvision import transforms

work_net = net

work_net.eval()
work_net.load_state_dict(torch.load('parameters_resnet_18_mnist.pth'))

path = 'C:\\Users\\marenkun\\Desktop\\picture1\\5.jpg'
img_array = dealimg(path)
img_tensor = torch.tensor(img_array, dtype=torch.float32)


img_tensor = img_tensor.reshape(1, 1, 28, 28)
# 创建一个转换对象，将图像大小调整为96x96像素
transform = transforms.Resize((96, 96), antialias=False)


# 应用转换到图像上
img_resized = transform(img_tensor)
print(img_resized.shape)


# 如果模型在 GPU 上，需要将图像张量也移动到 GPU
if torch.cuda.is_available():
    work_net = work_net.cuda()
    img_resized = img_resized.cuda()

# 不需要计算梯度，使用 torch.no_grad() 上下文管理器
with torch.no_grad():
    output = work_net(img_resized)
# output 现在包含了模型的输出

# 首先，将输出张量从GPU转移到CPU
output_cpu = output.cpu()

# 找到得分最高的类别的索引
_, predicted_index = torch.max(output_cpu, 1)

# 获取索引值作为Python整数
predicted_digit = predicted_index.item()

# 输出预测的数字
print("Predicted digit:", predicted_digit)