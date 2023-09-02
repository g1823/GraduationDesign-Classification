import numpy as np
import torch
from PIL import Image
import requests
from io import BytesIO


def preprocess_image(pic_URL):
    # 加载图像
    url = pic_URL
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    # 将图像大小调整为VGG16所需的大小（224x224）
    image = image.resize((224, 224))
    # 将图像转换为Numpy数组，并进行预处理
    image = np.array(image)
    if image.ndim == 2:
        # 如果图像是灰度图，将其转换为RGB格式
        image = np.tile(np.expand_dims(image, 2), [1, 1, 3])
    elif image.shape[2] == 4:
        # 如果图像有Alpha通道，去掉Alpha通道
        image = image[:, :, :3]
    image = image.astype('float32')
    image = image / 255.0
    image = image - [0.485, 0.456, 0.406]
    image = image / [0.229, 0.224, 0.225]
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = torch.from_numpy(image).type(torch.cuda.FloatTensor)
    return image
