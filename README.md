# ISIC2018 Task1 皮肤病变分割 - 代码说明文档

## 一、项目概述

本项目基于U-Net深度学习模型，对ISIC2018数据集中的皮肤镜图像进行病变区域自动分割。

## 二、文件结构说明

```
ISIC2018_Task1/
├── data/                          # 数据集目录（需自行放入）
│   ├── ISIC2018_Task1-2_Training_Input/       # 训练集图片
│   ├── ISIC2018_Task1_Training_GroundTruth/   # 训练集标签
│   ├── ISIC2018_Task1-2_Validation_Input/     # 验证集图片
│   ├── ISIC2018_Task1_Validation_GroundTruth/ # 验证集标签
│   ├── ISIC2018_Task1-2_Test_Input/           # 测试集图片
│   └── ISIC2018_Task1_Test_GroundTruth/       # 测试集标签
├── checkpoints/                   # 模型权重保存目录
├── predictions/                   # 预测结果输出目录
├── model.py                       # U-Net模型定义
├── dataset.py                     # 数据集加载与预处理
├── train.py                       # 模型训练脚本
├── predict.py                     # 模型测试与推理脚本
├── utils.py                       # 工具函数（损失函数、评估指标）
├── requirements.txt               # 依赖包列表
└── README.md                      # 本说明文档
```

## 三、各文件功能详解

### model.py - 模型定义
- **UNet类**：标准U-Net架构，包含编码器（下采样路径）和解码器（上采样路径）
- **DoubleConv类**：双层卷积模块，每层包含 Conv2d → BatchNorm → ReLU
- 编码器特征通道数：[64, 128, 256, 512]
- 瓶颈层通道数：1024
- 输入：3通道RGB图像（256×256）
- 输出：1通道分割概率图（256×256）

### dataset.py - 数据集加载
- **ISIC2018Dataset类**：继承PyTorch Dataset
- 训练模式下包含数据增强：随机翻转、旋转、颜色抖动
- 验证/测试模式仅进行尺寸调整和归一化
- 图像归一化参数：ImageNet均值和标准差

### train.py - 训练脚本
- 使用Adam优化器，初始学习率1e-4，权重衰减1e-4
- 学习率调度：ReduceLROnPlateau（patience=5，factor=0.5）
- 损失函数：Dice Loss + BCE Loss组合
- 自动保存最佳模型和最新模型

### predict.py - 测试推理
- 支持单张图片预测和批量测试集评估
- 输出Dice系数、IoU、像素精度三项指标
- 自动保存预测分割结果图

### utils.py - 工具函数
- **DiceBCELoss**：组合损失函数，平衡像素级和区域级优化
- **dice_coefficient**：Dice相似系数，衡量预测与真值重叠度
- **iou_score**：交并比，衡量分割区域准确性
- **pixel_accuracy**：像素级分类精度

## 四、参数设置说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| img_size | 256 | 输入图像尺寸 |
| batch_size | 8 | 批次大小 |
| epochs | 50 | 训练轮数 |
| lr | 1e-4 | 初始学习率 |
| weight_decay | 1e-4 | 权重衰减（L2正则化） |

## 五、模型执行流程

### 环境安装
```bash
pip install -r requirements.txt
```

### 数据准备
将ISIC2018数据集解压到 `data/` 目录下，保持原始文件夹命名。

### 模型训练
```bash
python train.py --epochs 50 --batch_size 8 --lr 1e-4
```

### 模型测试
```bash
python predict.py --model_path checkpoints/best_model.pth
```

### 单张图片预测
```bash
python predict.py --model_path checkpoints/best_model.pth --single_image path/to/image.jpg
```

## 六、预训练模型权重

本项目从头训练U-Net模型，未使用预训练权重。训练完成后权重保存在：
- `checkpoints/best_model.pth`：验证集Dice最高的模型
- `checkpoints/last_model.pth`：最后一个epoch的模型

## 七、实验中遇到的问题及解决方案

1. **显存不足**：将batch_size从16减小到8，或将img_size从512减小到256
2. **训练不收敛**：使用Dice+BCE组合损失替代单一BCE损失，提升对类别不平衡的鲁棒性
3. **过拟合**：加入数据增强（随机翻转、旋转、颜色抖动）和权重衰减正则化
4. **分割边缘不清晰**：使用BatchNorm稳定训练，skip connection保留细节信息

## 八、模型完成情况

- [x] U-Net模型实现
- [x] 数据加载与增强
- [x] 模型训练流程
- [x] 模型评估与预测
- [x] 评估指标计算（Dice、IoU、Accuracy）
