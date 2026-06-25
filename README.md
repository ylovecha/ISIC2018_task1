# ISIC2018 Task1 皮肤病变分割 - 代码说明文档

## 一、项目概述

本项目基于深度学习语义分割模型，对 ISIC2018 数据集中的皮肤镜图像进行病变区域自动分割。实现了两种模型：

- **基础 U-Net**：标准编码器-解码器结构，从零训练
- **ResNet34-UNet（默认）**：ImageNet 预训练 ResNet34 编码器 + U-Net 风格解码器

在完整 ISIC2018 数据集（训练 2,594 / 验证 100 / 测试 1,000）上，ResNet34-UNet 测试集 Dice 达到 **0.8903**。

## 二、文件结构说明

```
ISIC2018_task1/
├── data/                              # 数据集目录（需自行放入）
│   ├── ISIC2018_Task1-2_Training_Input/       # 训练集图片
│   ├── ISIC2018_Task1_Training_GroundTruth/   # 训练集标签
│   ├── ISIC2018_Task1-2_Validation_Input/     # 验证集图片
│   ├── ISIC2018_Task1_Validation_GroundTruth/ # 验证集标签
│   ├── ISIC2018_Task1-2_Test_Input/           # 测试集图片
│   └── ISIC2018_Task1_Test_GroundTruth/       # 测试集标签（评估必需）
├── checkpoints/                       # ResNet34-UNet 权重（默认）
├── checkpoints_unet/                  # 基础 U-Net 权重（建议单独保存）
├── predictions/                       # 预测结果输出目录
├── pic/                               # 训练曲线与可视化图片
├── model.py                           # U-Net 模型定义
├── model_resnet_unet.py               # ResNet34-UNet 模型定义
├── dataset.py                         # 数据集加载与预处理
├── train.py                           # 模型训练脚本
├── predict.py                         # 模型测试与推理脚本
├── utils.py                           # 工具函数（损失函数、评估指标）
├── requirements.txt                   # 依赖包列表
├── 实验报告.md                         # 实验报告
└── README.md                          # 本说明文档
```

## 三、各文件功能详解

### model.py - 基础 U-Net

- **UNet 类**：标准 U-Net 架构，包含编码器（下采样）和解码器（上采样）
- **DoubleConv 类**：双层卷积模块（Conv2d → BatchNorm → ReLU）
- 编码器通道：[64, 128, 256, 512]，瓶颈层 1024，参数量约 31M
- 输入：3 通道 RGB（256×256），输出：1 通道分割 logits（256×256）

### model_resnet_unet.py - ResNet34-UNet

- **ResNetUNet 类**：torchvision ResNet34（ImageNet 预训练）编码器 + U-Net 解码器
- 跳跃连接来自 ResNet 各层特征，参数量约 29.2M
- 训练时默认 `pretrained=True`；推理加载 checkpoint 时使用 `pretrained=False`

### dataset.py - 数据集加载

- **ISIC2018Dataset 类**：继承 PyTorch Dataset
- 训练模式：随机翻转、旋转（±20°）、颜色抖动 + ImageNet 归一化
- 验证/测试模式：仅 Resize 与归一化
- 标签命名：`{图像名}_segmentation.png`

### train.py - 训练脚本

- 支持 `--model unet` 或 `--model resnet_unet`（默认）
- Adam 优化器（lr=1e-4，weight_decay=1e-4）
- ReduceLROnPlateau（patience=5，factor=0.5）
- Dice + BCE 组合损失，AMP 混合精度（CUDA 默认开启）
- 自动保存 `best_model.pth`（验证 Dice 最高）和 `last_model.pth`

### predict.py - 测试推理

- 根据 checkpoint 中 `model_type` 自动加载对应模型
- 批量测试集评估：输出 Dice、IoU、Pixel Accuracy
- 保存测试集前 20 张预测图至 `predictions/`

### utils.py - 工具函数

- **DiceBCELoss**：BCE + Dice 组合损失
- **dice_coefficient / iou_score / pixel_accuracy**：评估指标（sigmoid + 0.5 阈值）

## 四、参数设置说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| model | resnet_unet | 模型架构：`unet` 或 `resnet_unet` |
| img_size | 256 | 输入图像尺寸 |
| batch_size | 8 | 批次大小 |
| epochs | 50 | 训练轮数 |
| lr | 1e-4 | 初始学习率 |
| weight_decay | 1e-4 | 权重衰减（L2 正则化） |
| save_dir | checkpoints | 模型保存目录 |

## 五、模型执行流程

### 环境安装

```bash
pip install -r requirements.txt
```

### 数据准备

将 ISIC2018 数据集解压到 `data/` 目录，确保各子目录下直接包含 `.jpg` 图片和 `_segmentation.png` 标签（不要多嵌套一层）。

检查数据是否就绪：

```bash
ls data/ISIC2018_Task1-2_Training_Input/*.jpg | wc -l    # 期望 2594
ls data/ISIC2018_Task1-2_Validation_Input/*.jpg | wc -l  # 期望 100
ls data/ISIC2018_Task1-2_Test_Input/*.jpg | wc -l        # 期望 1000
ls data/ISIC2018_Task1_Test_GroundTruth/*_segmentation.png | wc -l  # 期望 1000
```

### 模型训练

训练 ResNet34-UNet（默认，推荐）：

```bash
python train.py --model resnet_unet --epochs 50 --batch_size 8 --lr 1e-4
```

训练基础 U-Net（建议指定独立保存目录）：

```bash
python train.py --model unet --epochs 50 --batch_size 8 --lr 1e-4 --save_dir checkpoints_unet
```

### 模型测试

ResNet34-UNet：

```bash
python predict.py --model_path checkpoints/best_model.pth
```

基础 U-Net：

```bash
python predict.py --model_path checkpoints_unet/best_model.pth
```

### 单张图片预测

```bash
python predict.py --model_path checkpoints/best_model.pth --single_image path/to/image.jpg
```

## 六、实验结果（实测）

在完整 ISIC2018 数据集上训练 50 epoch 后的结果：

| 指标 | 基础 U-Net | ResNet34-UNet | 提升 |
|------|-----------|---------------|------|
| 验证集最佳 Dice | 0.8859（Ep.44） | **0.9041**（Ep.36） | +2.05% |
| 测试集 Dice | 0.8757 | **0.8903** | +1.67% |
| 测试集 IoU | 0.7994 | **0.8171** | +2.21% |
| 测试集 Pixel Accuracy | 0.9304 | **0.9385** | +0.87% |

训练完成后权重保存在：

- `checkpoints/best_model.pth`：ResNet34-UNet 验证集 Dice 最高模型
- `checkpoints_unet/best_model.pth`：基础 U-Net 验证集 Dice 最高模型
- `checkpoints/last_model.pth`：最后一个 epoch 的模型

## 七、实验中遇到的问题及解决方案

1. **显存不足**：将 batch_size 从 16 减小到 8，或将 img_size 从 512 减小到 256
2. **训练不收敛**：使用 Dice+BCE 组合损失替代单一 BCE 损失，缓解类别不平衡
3. **过拟合**：加入数据增强（翻转、旋转、颜色抖动）和权重衰减正则化
4. **验证集样本数为 0**：验证集图片目录为空时会报错；检查路径或临时指定 `--val_mask_dir data/__no_val__` 触发训练集 90/10 划分
5. **测试集 Dice 为 0**：测试集 Ground Truth 未正确放置，mask 全部读为空；确认 `ISIC2018_Task1_Test_GroundTruth` 下有 1000 张 `_segmentation.png`
6. **Windows 多进程**：DataLoader 需将训练脚本置于 `if __name__ == '__main__'` 保护下
7. **GPU 利用率低**：设置 `num_workers=2` 并启用 `pin_memory`

## 八、模型完成情况

- [x] U-Net 模型实现
- [x] ResNet34-UNet 改进模型实现
- [x] 数据加载与增强
- [x] 模型训练流程（双模型、AMP）
- [x] 模型评估与预测
- [x] 评估指标计算（Dice、IoU、Accuracy）
- [x] 实验报告与训练曲线可视化
