import torch
from wrn28_10 import WideResNet

# 准备输入数据，这里以图像数据为例
input_data = torch.randn(1, 3, 32, 32)  # 1表示批量大小，3表示图像通道数，32表示图像尺寸
print(input_data)
print("----------------------")

# 创建 WideResNet 模型实例
model = WideResNet(depth=28, widen_factor=10, num_classes=10)

# 将模型设置为评估模式
model.eval()

# 使用模型获取输入数据的向量
output = model(input_data)

# 打印输出向量
print(output)
