# from DiffJPEG import DiffJPEG
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from modules.compression import compress_jpeg

image = Image.open('/home/dell/experiment/DeepFakesDefense/result/ori_jpg/000669.jpg')
transform = transforms.Compose([
    transforms.ToTensor(),
])
tensor = transform(image)
tensor = tensor.unsqueeze(0)

compress = compress_jpeg()
y, cb, cr = compress(tensor)
print(type(y))
print(y.shape)

# jpeg = DiffJPEG(height=256, width=256, differentiable=True, quality=80)
# result = jpeg(tensor)
# result = result.squeeze()

# array = result.permute(1, 2, 0).detach().numpy()

# plt.imshow(array)
# plt.axis('off')
# plt.show()