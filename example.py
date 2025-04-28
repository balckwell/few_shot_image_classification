import os
import argparse
import torch
import numpy as np
from PIL import Image
import clip11 as clip
from torchvision import transforms
from model import CLIPFewShotModel
from data_utils import load_image_paths_and_labels, get_few_shot_loaders

def run_few_shot_example(support_dir, query_dir, clip_model_path, classifier_model_path, finetuned_layers_path, output_file):
    """
    运行few-shot学习示例
    
    Args:
        support_dir: 支持集目录
        query_dir: 查询集目录
        clip_model_path: CLIP模型路径
        classifier_model_path: 分类器模型路径
        finetuned_layers_path: 微调层参数路径
        output_file: 输出文件路径
    """
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 加载CLIP模型
    print("加载CLIP模型...")
    model, preprocess = clip.load(clip_model_path, device=device)
    model = model.float()
    
    # 加载完整的few-shot模型
    print("加载Few-Shot模型...")
    few_shot_model = CLIPFewShotModel.load_from_checkpoint(
        classifier_model_path, 
        finetuned_layers_path, 
        model,
        num_classes=649,  # 与训练时相同
        num_unfreeze_blocks=6
    )
    
    # 创建数据加载器
    print("准备数据...")
    support_loader, query_loader, label_to_idx = get_few_shot_loaders(
        support_dir, 
        query_dir, 
        preprocess,
        batch_size=32,
        num_workers=4
    )
    
    # 创建idx_to_label映射
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    # 提取支持集特征
    print("提取支持集特征...")
    support_features, support_labels = few_shot_model.extract_features(support_loader)
    
    # 计算原型
    print("计算类别原型...")
    prototypes = few_shot_model.compute_prototypes(support_features, support_labels)
    
    # 提取查询集特征
    print("提取查询集特征...")
    query_features, _ = few_shot_model.extract_features(query_loader)
    
    # 进行分类
    print("执行Few-Shot分类...")
    predicted_labels, similarities = few_shot_model.classify_with_prototypes(
        query_features, 
        prototypes, 
        idx_to_label
    )
    
    # 获取查询集图像路径
    query_image_paths, _ = load_image_paths_and_labels(query_dir, is_support=False)
    
    # 保存结果
    print("保存分类结果...")
    results = ['img_name,label']
    for img_path, label in zip(query_image_paths, predicted_labels):
        img_name = os.path.basename(img_path)
        results.append(f"{img_name},{label}")
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(results))
    
    print(f"Few-Shot分类结果已保存到: {output_file}")
    
    # 打印分类准确率（如果有真实标签）
    # 此处无法获取真实标签，仅作为示例
    print("\n分类结果示例:")
    for i in range(min(5, len(query_image_paths))):
        img_name = os.path.basename(query_image_paths[i])
        prediction = predicted_labels[i]
        confidence = np.max(similarities[i])
        print(f"图像: {img_name}, 预测类别: {prediction}, 置信度: {confidence:.4f}")

def parse_args():
    parser = argparse.ArgumentParser(description='Few-Shot学习示例')
    parser.add_argument('--support_dir', type=str, required=True, help='支持集目录')
    parser.add_argument('--query_dir', type=str, required=True, help='查询集目录')
    parser.add_argument('--clip_model', type=str, default='ViT-B-32.pt', help='CLIP模型路径')
    parser.add_argument('--classifier_model', type=str, default='classifier_model.pth', help='分类器模型路径')
    parser.add_argument('--finetuned_layers', type=str, default='finetuned_layers.pth', help='微调层参数路径')
    parser.add_argument('--output', type=str, default='predictions.csv', help='输出文件路径')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_few_shot_example(
        args.support_dir,
        args.query_dir,
        args.clip_model,
        args.classifier_model,
        args.finetuned_layers,
        args.output
    )