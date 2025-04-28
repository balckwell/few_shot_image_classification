import os
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import clip as clip
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
import logging
import random
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 获取当前脚本的绝对路径
__parent_directory__ = os.path.abspath(os.path.dirname(__file__))

# 分类器模型
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        # 第一个全连接层
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )
        # 分类层
        self.fc2 = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        # 残差连接
        original_x = x
        # 主分支的前向传播
        x = self.fc1(x) 
        # 添加残差连接
        x += original_x  
        # 最后一层分类
        x = self.fc2(x)
        return x

class TrainingDataset(Dataset):
    def __init__(self, data_root, transform=None, is_training=True):
        self.data_root = data_root
        self.transform = transform
        self.is_training = is_training
        self.images = []
        self.labels = []
        self.class_map = {}
        
        # 加载数据
        self._load_data()
        
    def _load_data(self):
        class_folders = [d for d in os.listdir(self.data_root) if os.path.isdir(os.path.join(self.data_root, d))]
        
        for i, class_name in enumerate(class_folders):
            self.class_map[class_name] = i
            class_path = os.path.join(self.data_root, class_name)
            
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('png', 'jpg', 'jpeg')):
                    img_path = os.path.join(class_path, img_name)
                    self.images.append(img_path)
                    self.labels.append(i)
                    
        logger.info(f"加载了 {len(self.images)} 张图片, {len(class_folders)} 个类别")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            logger.error(f"加载图片 {img_path} 时出错: {e}")
            # 返回一个随机的其他样本
            return self.__getitem__(random.randint(0, len(self.images) - 1))

def train(args):
    # 设置随机种子以确保可重复性
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # 加载预训练模型
    logger.info("加载CLIP模型...")
    model_path = os.path.join(__parent_directory__, 'ViT-B-32.pt')
    model, preprocess = clip.load(model_path, device=device)
    model = model.float()
    visual_model = model.visual
    
    # 解冻最后 N 个 Transformer Block
    N = args.unfreeze_blocks
    for param in visual_model.parameters():
        param.requires_grad = False
    
    # 只解冻最后N个block
    finetuned_layers_dict = {}
    for idx in range(-N, 0):
        block = visual_model.transformer.resblocks[idx]
        for param in block.parameters():
            param.requires_grad = True
        # 收集每个block的参数以便后续保存
        finetuned_layers_dict[f'transformer.resblocks.{idx}'] = {k: v for k, v in block.state_dict().items()}
    
    logger.info(f"解冻最后 {N} 个Transformer Block进行微调")
    
    # 准备数据增强和预处理
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    
    val_transform = preprocess
    
    # 加载数据集
    train_dataset = TrainingDataset(args.train_data, transform=train_transform)
    val_dataset = TrainingDataset(args.val_data, transform=val_transform, is_training=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 初始化分类器
    num_classes = len(train_dataset.class_map)
    feature_dim = visual_model.output_dim
    classifier = Classifier(feature_dim, num_classes).to(device)
    
    # 设置优化器
    visual_params = [p for p in visual_model.parameters() if p.requires_grad]
    classifier_params = classifier.parameters()
    
    params = [
        {'params': visual_params, 'lr': args.visual_lr},
        {'params': classifier_params, 'lr': args.classifier_lr}
    ]
    
    optimizer = optim.AdamW(params, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()  # 用于混合精度训练
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 训练循环
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        visual_model.train()
        classifier.train()
        
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # 混合精度训练
            with autocast():
                features = visual_model(images)
                outputs = classifier(features)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            progress_bar.set_postfix({
                'loss': train_loss / (progress_bar.n + 1),
                'acc': 100. * train_correct / train_total
            })
        
        # 每个epoch结束后进行验证
        visual_model.eval()
        classifier.eval()
        
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images, labels = images.to(device), labels.to(device)
                
                with autocast():
                    features = visual_model(images)
                    outputs = classifier(features)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        logger.info(f"Epoch {epoch+1}/{args.epochs} - "
                   f"Train Loss: {train_loss / len(train_loader):.4f}, "
                   f"Train Acc: {100. * train_correct / train_total:.2f}%, "
                   f"Val Loss: {val_loss / len(val_loader):.4f}, "
                   f"Val Acc: {val_acc:.2f}%")
        
        # 更新学习率
        scheduler.step()
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            logger.info(f"保存最佳模型，验证准确率: {val_acc:.2f}%")
            
            # 保存微调的层参数
            finetuned_layers_state_dict = {}
            for idx in range(-N, 0):
                block = visual_model.transformer.resblocks[idx]
                finetuned_layers_state_dict[f'transformer.resblocks.{idx}'] = block.state_dict()
            
            torch.save(finetuned_layers_state_dict, os.path.join(__parent_directory__, 'finetuned_layers.pth'))
            torch.save(classifier.state_dict(), os.path.join(__parent_directory__, 'classifier_model.pth'))
    
    logger.info(f"训练完成! 最佳验证准确率: {best_val_acc:.2f}%")
    return visual_model, classifier

def parse_args():
    parser = argparse.ArgumentParser(description='CLIP微调训练')
    parser.add_argument('--train_data', type=str, required=True, help='训练数据目录')
    parser.add_argument('--val_data', type=str, required=True, help='验证数据目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--visual_lr', type=float, default=1e-5, help='视觉模型学习率')
    parser.add_argument('--classifier_lr', type=float, default=5e-4, help='分类器学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--unfreeze_blocks', type=int, default=6, help='解冻的transformer block数量')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)