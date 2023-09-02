import os
import random
import shutil

# 将原始数据集的每个类别分别划分为训练集和测试集，并存储到相应的文件夹中
root_dir = "dataset/Caltech256/256_ObjectCategories"
train_dir = "dataset/Caltech256/train"
test_dir = "dataset/Caltech256/test"

# 首先，代码检查训练集和测试集的目录是否存在，如果不存在，则创建相应的目录。
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# 对于数据集中的每个类别，分别创建训练集和测试集的目录。
for class_name in os.listdir(root_dir):
    class_dir = os.path.join(root_dir, class_name)
    train_class_dir = os.path.join(train_dir, class_name)
    test_class_dir = os.path.join(test_dir, class_name)
    if not os.path.exists(train_class_dir):
        os.makedirs(train_class_dir)
    if not os.path.exists(test_class_dir):
        os.makedirs(test_class_dir)
    # 对于每个类别的每个文件，代码使用随机数将其分配到训练集或测试集中，并将文件从原始目录复制到相应的训练集或测试集目录中。
    for file_name in os.listdir(class_dir):
        if random.random() < 0.8:
            shutil.copyfile(os.path.join(class_dir, file_name), os.path.join(train_class_dir, file_name))
        else:
            shutil.copyfile(os.path.join(class_dir, file_name), os.path.join(test_class_dir, file_name))