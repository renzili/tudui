import torchvision
from torch.utils.data import DataLoader

import time

from torch.utils.tensorboard import SummaryWriter

from modei import *
import wandb
wandb.init(project="tudui_loss and acc")

train_data = torchvision.datasets.CIFAR10(root="./torchvision_dataset", train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="./torchvision_dataset", train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)

train_data_size = len(train_data)  # 包含多少张图片
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 利用Dataloader来加载数据集

train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
tudui = Tudui()
if torch.cuda.is_available():
    tudui = tudui.cuda()

# 损失函数
Loss_fn = nn.CrossEntropyLoss()  # 交叉熵
if torch.cuda.is_available():
    loss_fn = Loss_fn.cuda()

# 优化器
learning_rate = 0.01  # 或者 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)  # 随机梯度下降

# 设置训练网络的一些参数
# 记录训练次数
total_train_step = 0
# 记录测试次数
total_test_step = 0
# 训练的次数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("logs")
record_train_loss, record_train_step = [], []
record_test_loss, record_test_step, record_test_acc = [], [], []


start_time = time.time()
for i in range(epoch):
    print("-----第 {} 轮训练开始-----".format(i + 1))
    # 训练步骤开始
    tudui.train()
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = tudui(imgs)
        loss = Loss_fn(outputs, targets)

        # 优化器调优
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  # 对参数进行优化
        # 记录训练次数
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print("训练次数：{}, loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

            record_train_loss.append(loss)
            record_train_step.append(total_train_step)

            data_train_loss = [[x, y] for (x, y) in zip(record_train_step, record_train_loss)]
            table1 = wandb.Table(data=data_train_loss, columns=["x", "y"])
            line_plot1 = wandb.plot.line(table1, "x", "y", title="train_loss")
            wandb.log({"train_loss": line_plot1})

    # 测试步骤开始
    tudui.eval()
    total_test_loss = 0
    x_y = 0

    total_accuracy = 0
    with torch.no_grad():  # 无梯度，保证不会进行调优操作，，是为了使用测试集检验训练结果
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = tudui(imgs)
            loss = Loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss  # 整个数据集的loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step, )
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step, )



    # data_test_acc = [[x, y] for (x, y) in zip(range(1,11), total_accuracy / test_data_size)]
    # table3 = wandb.Table(data=data_test_acc, columns=["x", "y"])
    # line_plot3 = wandb.plot.line(table3, "x", "y", title="test_acc")
    # wandb.log({"test_acc": line_plot3})

    total_test_step = total_test_step + 1

    record_test_loss.append(total_test_loss)
    record_test_step.append(total_test_step)
    x_y = total_accuracy / test_data_size
    record_test_acc.append(x_y)

    data_test_loss = [[x, y] for (x, y) in zip(record_test_step, record_test_loss)]
    table2 = wandb.Table(data=data_test_loss, columns=["x", "y"])
    line_plot2 = wandb.plot.line(table2, "x", "y", title="test_loss")
    wandb.log({"test_loss": line_plot2})

    data_test_acc = [[x, y] for (x, y) in zip(record_test_step, record_test_acc)]
    table3 = wandb.Table(data=data_test_acc, columns=["x", "y"])
    line_plot3 = wandb.plot.line(table3, "x", "y", title="test_acc")
    wandb.log({"test_acc": line_plot3})

    torch.save(tudui, "tudui_{}.pth".format(i))
    # trrch.save(tudui.state_dict() , “tudui_{}.pth”.format(i))   保存方式2
    print("模型已保存")

writer.close()
