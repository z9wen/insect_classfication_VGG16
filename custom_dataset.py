from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, txt_path, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.data = []
        with open(txt_path, 'r') as file:
            for line in file:
                path, label = line.strip().split()
                self.data.append((path, int(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, label = self.data[index]
        image = Image.open(os.path.join(self.img_dir, img_path))
        if self.transform:
            image = self.transform(image)
            # 打印看看是否有超出范围的标签
        if label < 0 or label > 102:
            print(f"标签超出范围：{label} at index {index}")
        return image, label
# import torch
# from torchvision import transforms
# from torch.utils.data import Dataset
# from PIL import Image
# import os
# from torch.nn.functional import one_hot
# class CustomDataset(Dataset):
#     def __init__(self, txt_path, img_dir, transform=None, num_classes=102):
#         self.img_labels = []  # 这里应该解析txt文件获取图片路径和标签
#         with open(txt_path, 'r') as file:
#             for line in file:
#                 path, label = line.strip().split()
#                 self.img_labels.append((path, int(label)))
#         self.img_dir = img_dir
#         self.transform = transform
#         self.num_classes = num_classes
#
#     def __len__(self):
#         return len(self.img_labels)
#
#     def __getitem__(self, idx):
#         img_path, label = self.img_labels[idx]
#         img = Image.open(os.path.join(self.img_dir, img_path)).convert('RGB')
#         label = one_hot(torch.tensor(label), num_classes=self.num_classes)  # 转换为one-hot编码
#         if self.transform:
#             img = self.transform(img)
#         return img, label



