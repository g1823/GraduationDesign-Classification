import torch
import torchvision.models as models

# 获取预训练的 VGG16 模型
vgg16 = models.vgg16(pretrained=True)

# 冻结所有参数
for param in vgg16.parameters():
    param.requires_grad = False

# 修改最后一层,将输出变为256，即符合Caltech256数据集数据集
vgg16.classifier[-1] = torch.nn.Linear(in_features=4096, out_features=256)

# 保存模型
torch.save(vgg16, "vgg16_caltech256.pth")