import torch
from PIL import Image
import torchvision.transforms as transforms
from getMap import get_category_map, get_English_tag, get_Chinese_tag
from image_preprocessing import preprocess_image

# 得到标签集合
category_map = get_category_map()
English_tag = get_English_tag()
Chinese_tag = get_Chinese_tag()

# 加载设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load("vgg16_caltech256.pth")
model.to(device)
model.eval()

# 图片预处理
img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图片大小至224x224像素
    transforms.ToTensor(),  # 将图片转换为PyTorch张量类型
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 标准化处理
])

# 加载图片
pic_dir = "dataset/Caltech256/256_ObjectCategories/001.ak47/001_0001.jpg"
img = Image.open(pic_dir)
img_tensor = img_transforms(img)
# PyTorch中，VGG16模型的输入张量形状是(batch_size, channels, height, width)
# 使用unsqueeze(0)函数将输入图片的维度从(batch_size, channels, height, width)
# 变为(batch_size=1, channels, height, width)，即增加了一个batch_size为1的维度。
# 这样就可以将单张图片作为输入传递给模型进行推理。
image_tensor = img_tensor.unsqueeze(0).to(device)

# 预测
with torch.no_grad():
    output = model(image_tensor)
predicted_label = torch.argmax(output).item()

print(Chinese_tag.get(predicted_label))
print(category_map.get(predicted_label))
