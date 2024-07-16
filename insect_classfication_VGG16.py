# import torch
# import torch.nn as nn
# from torchvision import transforms, models
# from torch.utils.data import DataLoader
# from custom_dataset import CustomDataset  # 确保CustomDataset已正确实现
# from torchvision.models import vgg16, VGG16_Weights
# from torchvision.transforms import Lambda
#
# # 设定设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # 打印当前使用的设备
# print(f"Using device: {device}")
#
# # 如果使用的是CUDA设备，打印具体的GPU型号
# if device.type == 'cuda':
#     print(f"GPU Model: {torch.cuda.get_device_name(device.index)}")
# # 数据转换
# transform = transforms.Compose([
#     transforms.Lambda(lambda x: x.convert('RGB')),  # 确保图像为RGB
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])
#
# # 创建数据集
# train_dataset = CustomDataset(txt_path='train.txt', img_dir='images/', transform=transform)
# val_dataset = CustomDataset(txt_path='val.txt', img_dir='images/', transform=transform)
# test_dataset = CustomDataset(txt_path='test.txt', img_dir='images/', transform=transform)
#
# # 创建数据加载器
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
#
# # 初始化模型
# model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
# num_classes = 102
# model.classifier[6] = nn.Linear(4096, num_classes)
# model = model.to(device)
#
# # 损失函数和优化器
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
# # 训练模型
# num_epochs = 5
# for epoch in range(num_epochs):
#     print(f'Starting epoch {epoch + 1}/{num_epochs}')
#     total = 0
#     correct = 0
#     for batch_idx, (images, labels) in enumerate(train_loader):
#         images, labels = images.to(device), labels.to(device)
#
#         # 前向传播
#         outputs = model(images)
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
#         # 计算损失
#         loss = criterion(outputs, labels)
#
#         # 反向传播和优化
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         if (batch_idx + 1) % 10 == 0:  # 每10个batch打印一次
#             accuracy = 100 * correct / total
#             print(f'Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%')
#
#     # 每个epoch结束后重置准确率计算
#     total = 0
#     correct = 0
#
#     # 验证
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for images, labels in val_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#         print(f'Epoch {epoch+1}, Validation Accuracy: {100 * correct / total:.2f}%')
#
# # 测试
# model.eval()
# correct = 0
# total = 0
# with torch.no_grad():
#     for images, labels in test_loader:
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#     print(f'Accuracy on test set: {100 * correct / total:.2f}%')

# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from torchvision.models import vgg16, VGG16_Weights
# from custom_dataset import CustomDataset
# from cutmix_utils import cutmix
# from sparse_loss import sparse_loss  # 导入sparse_loss函数
# import time
# # 设定设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # 打印当前使用的设备
# print(f"Using device: {device}")
#
# # 如果使用的是CUDA设备，打印具体的GPU型号
# if device.type == 'cuda':
#     print(f"GPU Model: {torch.cuda.get_device_name(device.index)}")
#
# # 数据转换
# transform = transforms.Compose([
#     transforms.Lambda(lambda x: x.convert('RGB')),  # 确保图像为RGB
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])
#
# # 创建数据集
# train_dataset = CustomDataset(txt_path='train.txt', img_dir='images/', transform=transform)
# val_dataset = CustomDataset(txt_path='val.txt', img_dir='images/', transform=transform)
# test_dataset = CustomDataset(txt_path='test.txt', img_dir='images/', transform=transform)
#
# # 创建数据加载器
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)


# model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
# for param in model.features.parameters():  # 冻结预训练层
#     param.requires_grad = False
# model.classifier[6] = nn.Linear(4096, 102)  # 调整最后一层
# model = model.to(device)
#
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
# # 训练模型，包括 CutMix
# def train(model, train_loader, val_loader, num_epochs, alpha=1.0):
#     for epoch in range(num_epochs):
#         start_time = time.time()  # 确保 start_time 在循环开始时被设置
#         model.train()
#         total = correct = 0
#         for batch_idx, (images, labels) in enumerate(train_loader):
#             images, labels = images.to(device), labels.to(device)
#
#             if np.random.rand() < 0.5:  # 50% 概率应用 CutMix
#                 images, targets_a, targets_b, lam = cutmix(images, labels, alpha)
#                 outputs = model(images)
#                 loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
#             else:
#                 outputs = model(images)
#                 loss = criterion(outputs, labels)
#
#             # sparse_reg_loss = sparse_loss(model, images)  # 计算sparse loss
#             # total_loss = regular_loss + 0.01 * sparse_reg_loss  # 组合损失
#
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             if (batch_idx + 1) % 10 == 0:
#                 accuracy = 100 * correct / total
#                 print(f'Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%')
#
#         elapsed_time = time.time() - start_time
#         print(f'Epoch {epoch + 1} completed in {elapsed_time:.2f} seconds')
#
#         # 验证模型性能
#         model.eval()
#         val_total = val_correct = 0
#         with torch.no_grad():
#             for images, labels in val_loader:
#                 images, labels = images.to(device), labels.to(device)
#                 outputs = model(images)
#                 _, predicted = torch.max(outputs, 1)
#                 val_total += labels.size(0)
#                 val_correct += (predicted == labels).sum().item()
#
#         val_accuracy = 100 * val_correct / val_total
#         print(f'Epoch {epoch + 1}, Validation Accuracy: {val_accuracy:.2f}%')
#
# # 调用函数
# train(model, train_loader, val_loader, num_epochs=10)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights
from custom_dataset import CustomDataset
from cutmix_utils import cutmix
from sparse_loss import sparse_loss  # 确保已导入sparse_loss函数
import time
from torch.optim.lr_scheduler import StepLR  # 导入学习率调度器
# 设定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据转换
transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert('RGB')),  # 确保图像为RGB
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 创建数据集
train_dataset = CustomDataset(txt_path='train.txt', img_dir='images/', transform=transform)
val_dataset = CustomDataset(txt_path='val.txt', img_dir='images/', transform=transform)
test_dataset = CustomDataset(txt_path='test.txt', img_dir='images/', transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# 初始化模型
model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
for param in model.features.parameters():  # 冻结预训练层
    param.requires_grad = False
model.avgpool = nn.AdaptiveAvgPool2d((7, 7))
model.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(512 * 7 * 7, 4096),
    nn.ReLU(True),
    nn.Dropout(0.2), # 添加Dropout层
    nn.Linear(4096, 4096),
    nn.ReLU(True),
    nn.Dropout(0.2), # 添加Dropout层
    nn.Linear(4096, 102)
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)

def train(model, train_loader, val_loader, num_epochs, alpha=1.0):
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=0.0001)  # 使用AdamW优化器
    scheduler= StepLR(optimizer, step_size=50, gamma=0.1)  # 学习率调度器
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        total = correct = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            if np.random.rand() < 0.5:  # 50% 概率应用 CutMix
                images, targets_a, targets_b, lam = cutmix(images, labels, alpha)
                outputs = model(images)
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            # 计算稀疏正则化损失
            reg_loss = sparse_loss(model, images)
            total_loss = loss + 0.01 * reg_loss  # 添加稀疏正则化损失

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 10 == 0:
                accuracy = 100 * correct / total
                print(f'Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {total_loss.item():.4f}, Accuracy: {accuracy:.2f}%')

        elapsed_time = time.time() - start_time
        print(f'Epoch {epoch + 1} completed in {elapsed_time:.2f} seconds')

        scheduler.step()  # 更新学习率
        # 验证模型性能
        model.eval()
        val_total = val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        print(f'Epoch {epoch + 1}, Validation Accuracy: {val_accuracy:.2f}%')

train(model, train_loader, val_loader, num_epochs=5)
