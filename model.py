import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

class Classifier(nn.Module):
    """
    分类器模型，使用残差连接提高性能
    """
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


class CLIPFewShotModel:
    """
    基于CLIP的少样本学习模型
    """
    def __init__(self, clip_model, visual_model, classifier=None, num_unfreeze_blocks=6):
        self.clip_model = clip_model
        self.visual_model = visual_model
        self.classifier = classifier
        self.num_unfreeze_blocks = num_unfreeze_blocks
        
    @staticmethod
    def load_from_checkpoint(model_path, finetuned_layers_path, clip_model, num_classes=649, num_unfreeze_blocks=6):
        """
        从检查点加载模型
        
        Args:
            model_path: 分类器模型路径
            finetuned_layers_path: 微调层参数路径
            clip_model: CLIP模型
            num_classes: 类别数量
            num_unfreeze_blocks: 解冻的transformer block数量
            
        Returns:
            model: 加载后的模型
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        visual_model = clip_model.visual
        
        # 加载微调的层参数
        finetuned_layers_state_dict = torch.load(finetuned_layers_path, map_location=device)
        for idx in range(-num_unfreeze_blocks, 0):
            layer = visual_model.transformer.resblocks[idx]
            layer.load_state_dict(finetuned_layers_state_dict[f'transformer.resblocks.{idx}'])
        
        # 加载分类器
        feature_dim = visual_model.output_dim
        classifier = Classifier(feature_dim, num_classes).to(device)
        classifier.load_state_dict(torch.load(model_path, map_location=device))
        
        return CLIPFewShotModel(clip_model, visual_model, classifier, num_unfreeze_blocks)
    
    def extract_features(self, dataloader):
        """
        提取特征
        
        Args:
            dataloader: 数据加载器
            
        Returns:
            features: 特征向量
            labels: 标签（如果有）
        """
        self.visual_model.eval()
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="提取特征"):
                if isinstance(batch, list) or isinstance(batch, tuple):
                    images = batch[0].to(device)
                    labels = batch[1].numpy() if len(batch) > 1 else None
                else:
                    images = batch.to(device)
                    labels = None
                
                with autocast():
                    features = self.visual_model(images)
                    features = features / features.norm(dim=-1, keepdim=True)
                
                all_features.append(features.cpu().numpy())
                if labels is not None:
                    all_labels.extend(labels)
        
        all_features = np.concatenate(all_features, axis=0)
        return all_features, all_labels if all_labels else None
    
    def compute_prototypes(self, features, labels):
        """
        计算原型特征向量
        
        Args:
            features: 特征向量
            labels: 标签
            
        Returns:
            prototypes: 原型特征向量
        """
        labels = np.array(labels)
        unique_labels = np.unique(labels)
        prototypes = []
        
        for label in unique_labels:
            class_indices = np.where(labels == label)[0]
            class_features = features[class_indices]
            class_prototype = class_features.mean(axis=0)  # 计算平均特征向量作为原型
            prototypes.append(class_prototype)
        
        prototypes = np.vstack(prototypes)  # 将所有类别原型拼接成矩阵
        prototypes = prototypes / np.linalg.norm(prototypes, axis=1, keepdims=True)  # 归一化
        
        return prototypes
    
    def classify_with_prototypes(self, query_features, prototypes, idx_to_label=None):
        """
        使用原型进行分类
        
        Args:
            query_features: 查询特征向量
            prototypes: 原型特征向量
            idx_to_label: 索引到标签的映射
            
        Returns:
            predictions: 预测标签
            similarities: 相似度分数
        """
        query_features = query_features / np.linalg.norm(query_features, axis=1, keepdims=True)  # 归一化
        similarities = np.dot(query_features, prototypes.T)  # 计算余弦相似度
        predicted_indices = similarities.argmax(axis=1)  # 选择相似度最高的类别
        
        if idx_to_label:
            predicted_labels = [idx_to_label[idx] for idx in predicted_indices]
            return predicted_labels, similarities
        else:
            return predicted_indices, similarities
    
    def classify_with_classifier(self, dataloader):
        """
        使用分类器进行分类
        
        Args:
            dataloader: 数据加载器
            
        Returns:
            predictions: 预测标签
            confidences: 置信度分数
        """
        if self.classifier is None:
            raise ValueError("分类器未加载")
        
        self.visual_model.eval()
        self.classifier.eval()
        
        all_predictions = []
        all_confidences = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="分类中"):
                if isinstance(batch, list) or isinstance(batch, tuple):
                    images = batch[0].to(device)
                else:
                    images = batch.to(device)
                
                with autocast():
                    features = self.visual_model(images)
                    outputs = self.classifier(features)
                    probabilities = torch.softmax(outputs, dim=1)
                
                confidences, predictions = torch.max(probabilities, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
        
        return all_predictions, all_confidences