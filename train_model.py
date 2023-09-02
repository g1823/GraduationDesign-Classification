import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch import nn
import torch.optim as optim
import time

# 数据集路径
dataset_dir = "dataset/Caltech256"

# 预处理
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图片大小至224x224像素
    transforms.RandomHorizontalFlip(),  # 随机水平翻转图片，增加数据集的多样性。
    transforms.ToTensor(),  # 将图片转换为 PyTorch 中的张量（tensor）类型，以便能够输入到神经网络中进行训练。
    # 对图片进行标准化处理，使得每个像素的数值都集中在 0 周围，并具有相同的方差。这个过程可以提高模型的训练效果和准确性。
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
# test_transforms = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
# ])

# 加载数据集
train_data = datasets.ImageFolder(root=dataset_dir + '/256_ObjectCategories', transform=train_transforms)
# test_data = datasets.ImageFolder(root=dataset_dir + '/test', transform=test_transforms)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)

train_num = len(train_data)
# test_num = len(test_data)
print("训练数据集数量：{}".format(train_num))
# print("验证数据集数量：{}".format(test_num))

# 加载训练设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    print('使用GPU训练')
else:
    print('使用CPU训练')

# 加载模型
model = torch.load("vgg16_caltech256.pth")
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion.to(device)

# 设置训练网络的参数
epoch = 10  # 训练轮数

start_time = time.time()  # 开始时间
temp_time = start_time

# 开始训练
for i in range(epoch):
    print("--------------第{}轮训练开始：---------------".format(i + 21))
    # 将模型设置为训练模式
    model.train()
    # 初始化训练损失和正确率
    train_loss = 0.0
    train_acc = 0.0
    for data in train_loader:
        images, targets = data
        images = images.to(device)
        targets = targets.to(device)
        # 梯度清零
        optimizer.zero_grad()
        # 前向传播，计算损失
        outputs = model(images)  # 将数据放到网络中训练
        loss = criterion(outputs, targets)  # 用损失函数得到差异值
        # 优化模型，反向传播，更新模型参数
        loss.backward()
        optimizer.step()
        # 统计训练损失和正确率
        train_loss += loss.item() * images.size(0)
        preds = torch.max(outputs, 1)
        train_acc += (preds[1] == targets).sum().item()

    # 计算平均训练损失和正确率
    train_loss = train_loss / train_num
    train_acc = train_acc / train_num
    # 计算平均验证损失和正确率
    print("第{}轮训练平均损失值为{}".format(i + 21, train_loss))
    print("第{}轮训练正确率为{}".format(i + 21, train_acc))

    end_time = time.time()
    print("第{}轮训练用时{}秒".format(i + 21, end_time - temp_time))
    temp_time = end_time

    # 保存模型
    torch.save(model, "vgg16_caltech256.pth")
    print("本轮模型已保存")
    print("--------------第{}轮训练结束：---------------".format(i + 21))

end_time = time.time()  # 结束时间
print("共用时{}秒".format(end_time-start_time))

