#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 16:05:01 2024

@author: kalen
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1 数据转换和加载
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(1024),  # 随机裁剪并调整大小到 1024x1024
    transforms.RandomHorizontalFlip(),   # 随机水平翻转
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
])

transform_val = transforms.Compose([
    transforms.Resize(1024),  # 调整大小
    transforms.CenterCrop(1024),  # 中心裁剪
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
])

train_dataset = datasets.ImageFolder(root='/path/to/imagenet/train', transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=16, pin_memory=True)

val_dataset = datasets.ImageFolder(root='/path/to/imagenet/val', transform=transform_val)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=16, pin_memory=True)


# 2 初始化模型
model = ImageEncoderViT()

# 使用GPU加速（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# 3 选择损失函数和优化器
criterion = nn.CrossEntropyLoss()  # ImageNet 是一个分类任务
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)


# 4 训练函数
def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(dataloader)
    return epoch_loss

# 验证函数
def validate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total
    return epoch_loss, accuracy


# 5 训练和验证
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train_model(model, train_loader, criterion, optimizer, device)
    val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")


# 6 保存模型
torch.save(model.state_dict(), "vit_encoder_imagenet.pth")
