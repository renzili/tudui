import torch
from torch import nn
from torch.nn import Conv2d


# 搭建神经网络  一个10分类的网络
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),   # in_channels , out_channels , kernel_size , stride , padding
            nn.MaxPool2d(2),  # kernel_size
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    tudui = Tudui()
    input = torch.ones((64, 3, 32, 32))   # batch_size  三通道  32×32
    output = tudui(input)
    print(output.shape)   # 输出为torch_size([64 , 10])  这个的意思是返回64行数据，每行有10个数据，这10 个数据代表每一个图片在10个类别中的概率。
    print(list(tudui.named_children()))

