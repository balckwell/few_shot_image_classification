import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random

class FewShotDataset(Dataset):
    """
    数据集类，用于加载few-shot学习的支持集和查询集
    """
    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            if self.labels is not None:
                return image, self.labels[idx]
            else:
                return image
        except Exception as e:
            print(f"读取图片 {img_path} 时出错: {e}")
            # 返回一个随机的其他样本
            return self.__getitem__(random.randint(0, len(self.image_paths) - 1))
            
class ImageDataset(Dataset):
    """
    图像数据集类，支持数据增强，用于特征提取
    """
    def __init__(self, image_paths, transform, augment_times=1):
        self.image_paths = image_paths
        self.transform = transform
        self.augment_times = augment_times

    def __len__(self):
        return len(self.image_paths) * self.augment_times

    def __getitem__(self, idx):
        actual_idx = idx % len(self.image_paths)
        try:
            image = Image.open(self.image_paths[actual_idx]).convert('RGB')
            image = self.transform(image)
            return image
        except Exception as e:
            print(f"读取图片 {self.image_paths[actual_idx]} 时出错: {e}")
            # 返回一个随机的其他样本
            return self.__getitem__(random.randint(0, len(self.image_paths) - 1))

def load_image_paths_and_labels(image_folder, is_support=False):
    """
    加载图像路径和标签
    
    Args:
        image_folder: 图像文件夹路径
        is_support: 是否为支持集，如果是，则从子文件夹名称获取标签
    
    Returns:
        image_paths: 图像路径列表
        labels: 标签列表（如果is_support=False，则为空列表）
    """
    image_paths = []
    labels = []
    if not os.path.exists(image_folder):
        print(f"Warning: 目录 {image_folder} 不存在")
        return image_paths, labels

    if is_support:
        for label in os.listdir(image_folder):
            label_path = os.path.join(image_folder, label)
            if os.path.isdir(label_path):
                for img_file in os.listdir(label_path):
                    if img_file.lower().endswith(('png', 'jpg', 'jpeg')):
                        img_path = os.path.join(label_path, img_file)
                        image_paths.append(img_path)
                        labels.append(label)
    else:
        for img_file in os.listdir(image_folder):
            if img_file.lower().endswith(('png', 'jpg', 'jpeg')):
                img_path = os.path.join(image_folder, img_file)
                image_paths.append(img_path)
    return image_paths, labels

def create_dataloaders(train_dir, val_dir, preprocess, batch_size=32, num_workers=4):
    """
    创建训练和验证数据加载器
    
    Args:
        train_dir: 训练数据目录
        val_dir: 验证数据目录
        preprocess: 预处理函数
        batch_size: 批处理大小
        num_workers: 数据加载线程数
    
    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        class_to_idx: 类别到索引的映射
    """
    # 训练数据增强
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    
    # 获取所有类别
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    classes.sort()  # 确保顺序一致
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    # 读取训练图像
    train_images = []
    train_labels = []
    for cls_name in classes:
        cls_dir = os.path.join(train_dir, cls_name)
        for img_name in os.listdir(cls_dir):
            if img_name.lower().endswith(('png', 'jpg', 'jpeg')):
                img_path = os.path.join(cls_dir, img_name)
                train_images.append(img_path)
                train_labels.append(class_to_idx[cls_name])
    
    # 读取验证图像
    val_images = []
    val_labels = []
    for cls_name in classes:
        if not os.path.exists(os.path.join(val_dir, cls_name)):
            continue
        cls_dir = os.path.join(val_dir, cls_name)
        for img_name in os.listdir(cls_dir):
            if img_name.lower().endswith(('png', 'jpg', 'jpeg')):
                img_path = os.path.join(cls_dir, img_name)
                val_images.append(img_path)
                val_labels.append(class_to_idx[cls_name])
    
    # 创建数据集
    train_dataset = FewShotDataset(train_images, train_labels, transform=train_transform)
    val_dataset = FewShotDataset(val_images, val_labels, transform=preprocess)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, class_to_idx

def get_few_shot_loaders(support_dir, query_dir, preprocess, batch_size=32, num_workers=4):
    """
    创建few-shot支持集和查询集数据加载器
    
    Args:
        support_dir: 支持集目录
        query_dir: 查询集目录
        preprocess: 预处理函数
        batch_size: 批处理大小
        num_workers: 数据加载线程数
    
    Returns:
        support_loader: A DataLoader for support set
        query_loader: A DataLoader for query set
        label_to_idx: Label to index mapping
    """
    # 获取支持集图像和标签
    support_images, support_labels = load_image_paths_and_labels(support_dir, is_support=True)
    
    # 创建标签映射
    unique_labels = list(set(support_labels))
    unique_labels.sort()  # 保持顺序一致性
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    # 将字符串标签转换为索引
    support_label_indices = [label_to_idx[label] for label in support_labels]
    
    # 获取查询集图像
    query_images, _ = load_image_paths_and_labels(query_dir, is_support=False)
    
    # 创建数据集
    support_dataset = FewShotDataset(support_images, support_label_indices, transform=preprocess)
    query_dataset = FewShotDataset(query_images, transform=preprocess)
    
    # 创建数据加载器
    support_loader = DataLoader(support_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    query_loader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return support_loader, query_loader, label_to_idx