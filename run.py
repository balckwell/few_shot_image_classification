import os
import sys
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import clip as clip
from torchvision import transforms
from torch.cuda.amp import autocast

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 获取当前脚本的绝对路径
__parent_directory__ = os.path.abspath(os.path.dirname(__file__))

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
        # 如果维度不匹配，先对输入进行变换
        original_x = x
        # 主分支的前向传播
        x = self.fc1(x) 
        # 添加残差连接
        x += original_x  
        # 最后一层分类
        x = self.fc2(x)
        return x


def main(to_pred_dir, result_save_path):
    # 使用__parent_directory__来获取模型的绝对路径
    model_path = os.path.join(__parent_directory__, 'ViT-B-32.pt')
    model, preprocess = clip.load(model_path, device=device)
    model = model.float()
    visual_model = model.visual

    # 解冻最后 N 个 Transformer Block
    N = 6
    for param in visual_model.parameters():
        param.requires_grad = False
    for block in visual_model.transformer.resblocks[-N:]:
        for param in block.parameters():
            param.requires_grad = True

    # 加载微调的层参数
    finetuned_layers_path = os.path.join(__parent_directory__, 'finetuned_layers.pth')
    finetuned_layers_state_dict = torch.load(finetuned_layers_path, map_location=device)
    for idx in range(-N, 0):
        layer = visual_model.transformer.resblocks[idx]
        layer.load_state_dict(finetuned_layers_state_dict[f'transformer.resblocks.{idx}'])
    visual_model.eval()

    # 加载微调的分类器模型
    classifier_model_path = os.path.join(__parent_directory__, 'classifier_model.pth')
    num_train_classes = 649
    feature_dim = visual_model.output_dim
    classifier = Classifier(feature_dim, num_train_classes).to(device)
    classifier.load_state_dict(torch.load(classifier_model_path, map_location=device))
    classifier.eval()

    # 设定数据增强及预处理方式
    test_transform = preprocess
    support_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.95, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    # 处理待预测目录
    dirpath = os.path.abspath(to_pred_dir)  # 获取绝对路径
    filepath = os.path.join(dirpath, 'testB')
    task_lst = os.listdir(filepath)

    res = ['img_name,label']
    for task_name in task_lst:
        support_path = os.path.join(filepath, task_name, 'support')
        query_path = os.path.join(filepath, task_name, 'query')

        if not os.path.isdir(support_path) or not os.path.isdir(query_path):
            print(f"Warning: 目录不存在 - support: {support_path}, query: {query_path}")
            continue

        # 加载支持集图像路径和标签
        support_image_paths, support_labels = load_image_paths_and_labels(support_path, is_support=True)

        support_labels = np.array(support_labels)
        unique_labels = np.unique(support_labels)
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}

        # 提取支持集特征
        support_features = extract_features(support_image_paths, visual_model, test_transform)
        support_labels_indices = np.array([label_to_idx[label] for label in support_labels])

        # 计算每个类别的原型特征向量
        prototypes = []
        for idx in range(len(unique_labels)):
            class_indices = np.where(support_labels_indices == idx)[0]
            class_features = support_features[class_indices]
            class_prototype = class_features.mean(axis=0)  # 每个类别的平均特征向量
            prototypes.append(class_prototype)
        prototypes = np.vstack(prototypes)  # 将所有类别原型拼接成矩阵

        # 特征归一化处理
        prototypes = prototypes / np.linalg.norm(prototypes, axis=1, keepdims=True)

        # 提取查询集特征
        query_image_paths, _ = load_image_paths_and_labels(query_path, is_support=False)
        query_features = extract_features(query_image_paths, visual_model, test_transform)
        query_features = query_features / np.linalg.norm(query_features, axis=1, keepdims=True)

        # 计算查询集特征与类别原型的余弦相似度并进行分类
        similarities = np.dot(query_features, prototypes.T)  # 计算查询集和原型之间的相似度
        predicted_indices = similarities.argmax(axis=1)      # 选择相似度最高的类别
        predicted_labels = [idx_to_label[idx] for idx in predicted_indices]

        # 保存结果
        test_img_lst = [os.path.basename(p) for p in query_image_paths]
        for img_name, predicted_label in zip(test_img_lst, predicted_labels):
            res.append(f"{img_name},{predicted_label}")

    # 保存预测结果
    result_save_path = os.path.abspath(result_save_path)  # 使用绝对路径保存结果
    try:
        with open(result_save_path, 'w') as f:
            f.write('\n'.join(res))
        print(f"结果已成功保存到 {result_save_path}")
    except Exception as e:
        print(f"保存结果到 {result_save_path} 时出错: {e}")

def load_image_paths_and_labels(image_folder, is_support=False):
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

class ImageDataset(Dataset):
    def __init__(self, image_paths, transform, augment_times=1):
        self.image_paths = image_paths
        self.transform = transform
        self.augment_times = augment_times

    def __len__(self):
        return len(self.image_paths) * self.augment_times

    def __getitem__(self, idx):
        actual_idx = idx % len(self.image_paths)
        image = Image.open(self.image_paths[actual_idx]).convert('RGB')
        image = self.transform(image)
        return image

def extract_features(image_paths, model, transform, batch_size=64, augment_times=1):
    dataset = ImageDataset(image_paths, transform, augment_times=augment_times)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    features = []
    for images in tqdm(dataloader, desc="提取特征"):
        images = images.to(device)
        with torch.no_grad():
            with autocast():
                image_features = model(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            features.append(image_features.cpu().numpy())
    features = np.concatenate(features, axis=0)
    return features

if __name__ == "__main__":
    to_pred_dir = sys.argv[1]
    result_save_path = sys.argv[2]
    main(to_pred_dir, result_save_path)