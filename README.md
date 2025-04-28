# Few-Shot Image Classification with CLIP

基于CLIP (Contrastive Language-Image Pre-training) 的少样本图像分类项目，使用微调的视觉编码器和原型网络进行分类。

## 项目简介

本项目是一个基于 CLIP 模型的少样本图像分类系统，通过微调 CLIP 的视觉编码器，提取图像特征，然后结合原型学习 (Prototypical Networks) 方法实现少样本学习。该系统适用于在标签数据有限的情况下快速学习新类别，类似于人类的"一眼就能认出"的能力。

主要技术特点：
- 使用 OpenAI 的 CLIP 模型提取图像特征
- 只微调 CLIP 视觉编码器的最后 N 个 Transformer Block，保留大部分预训练知识
- 加入残差连接的分类器，提高训练稳定性
- 支持原型学习的 few-shot 分类方法
- 支持混合精度训练，加速训练过程

## 环境要求

```
Python >= 3.7
PyTorch >= 1.7.1
torchvision >= 0.8.2
numpy >= 1.20.0
Pillow >= 8.0.0
tqdm >= 4.56.0
clip >= 2.0.0  # CLIP的Python实现
```

所有依赖都列在 `requirements.txt` 文件中，可以通过以下命令安装：

```bash
pip install -r requirements.txt
```

## 项目结构

```
few_shot_image_classification/
├── run.py                 # 推理脚本
├── train.py               # 模型训练脚本
├── model.py               # 模型定义
├── data_utils.py          # 数据加载工具
├── download_clip.py       # CLIP模型下载脚本
├── requirements.txt       # 项目依赖
└── README.md              # 项目说明文档
```

## 快速开始

### 1. 下载预训练 CLIP 模型

```bash
python download_clip.py --model_name ViT-B/32
```

### 2. 训练模型

```bash
python train.py --train_data /path/to/train_data --val_data /path/to/val_data --batch_size 32 --epochs 30
```

主要参数说明：
- `--train_data`: 训练集目录，每个类别应有一个子文件夹
- `--val_data`: 验证集目录，结构同训练集
- `--batch_size`: 批处理大小
- `--epochs`: 训练轮数
- `--visual_lr`: 视觉模型学习率 (默认: 1e-5)
- `--classifier_lr`: 分类器学习率 (默认: 5e-4)
- `--unfreeze_blocks`: 解冻的transformer block数量 (默认: 6)

### 3. 进行预测

```bash
python run.py /path/to/test_directory /path/to/result.csv
```

## 数据格式

训练数据的目录结构应该如下：

```
train_data/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
```

测试数据的目录结构应该如下：

```
test_directory/
└── testB/
    ├── task1/
    │   ├── support/  # 支持集，每个类别一个文件夹
    │   │   ├── class1/
    │   │   │   ├── image1.jpg
    │   │   │   └── ...
    │   │   ├── class2/
    │   │   │   ├── image1.jpg
    │   │   │   └── ...
    │   │   └── ...
    │   └── query/   # 查询集，需要预测的图像
    │       ├── image1.jpg
    │       ├── image2.jpg
    │       └── ...
    ├── task2/
    │   ├── support/
    │   └── query/
    └── ...
```

## 技术细节

### 模型架构

1. **基础模型**：OpenAI CLIP的ViT-B/32版本
2. **特征提取**：使用CLIP的视觉编码器提取图像特征
3. **分类器**：带残差连接的两层全连接网络
4. **Few-shot分类**：使用支持集计算类别原型，然后用余弦相似度进行分类

### 训练策略

1. 冻结大部分CLIP预训练参数，只微调最后N个Transformer Block
2. 使用AdamW优化器和余弦退火学习率
3. 混合精度训练加速
4. 采用数据增强提高泛化能力

## 性能提升技巧

1. 增加训练数据的增强多样性可以提高模型泛化能力
2. 调整`unfreeze_blocks`参数可以在特征提取能力和计算效率之间取得平衡
3. 使用更大的批处理大小通常可以提高训练稳定性
4. 较低的视觉模型学习率和较高的分类器学习率通常效果更好

## 许可证

MIT

## 致谢

本项目基于以下开源项目：
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [PyTorch](https://pytorch.org/)
- [Torchvision](https://github.com/pytorch/vision) 