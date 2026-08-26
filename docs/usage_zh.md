# 代码使用说明

## 流程概览

本项目实现“影像输入 → 分割 → 组织定量与影像组学 → 机器学习评价”。

| 阶段 | 实现 |
|---|---|
| 输入 | 常规 DICOM 分量转换，或直接读取 Water、Fat、FF 三通道 NIfTI；按空间位置排列切片 |
| 分割 | 受试者级 80:20 划分；开发集五折；六级 3D 网络；Adam；Dice 与 IMAT 加权交叉熵 |
| 预测掩膜 | 开发集折外预测，独立测试集五折集成；保存模型及受试者来源记录 |
| 特征 | 原始空间的体积、比例、肌肉脂肪分数，以及 PyRadiomics first-order、3D shape、GLCM |
| 建模 | Patient_ID 连接临床和影像表；七种特征策略 × LR/RF/XGBoost；内外层均为五折重复两次 |
| 预处理与选择 | 缺失值填补、缩放、去冗余、稳定性筛选、L1 选择在对应训练折内拟合 |
| 输出 | 逐受试者概率、评价指标、ROC/校准/DCA、折外 SHAP、特征稳定性、拟合模型及配置 |

分割的开发/测试划分与机器学习交叉验证分别执行。临床表定义机器学习人群：
影像表可以包含全部 114 名分割受试者，机器学习使用临床表中的 107 人。

## 安装与运行

按 [安装文档](installation.md) 创建 Python 环境，并准备：

- `subjects.csv`：Patient_ID、water、fat、ff、manual_mask；每行一个受试者，后四列为影像路径。
- `clinical.csv`：Patient_ID、Diabetes_Status、Age、Sex、BMI；结局 0=NGT，1=糖代谢异常。

```bash
leg-mri study-run --subjects data/subjects.csv --clinical data/clinical.csv --config configs/study.yaml --work outputs/study --device cuda
```

已有特征表时可以从 `model-cv` 开始。各阶段命令、恢复运行方式和输出结构见
[完整流程](pipeline.md)。

## 方法与参数

### 影像组学

使用 **Water+Fat**。每个 Water 体积按均值和样本标准差做 z-score，再乘以 100；
Fat 保持原始强度；随后以固定 binWidth=25 离散化。
配置为 `modalities: [water, fat]`、`water_normalize_scale: 100.0`。
FF 图用于分割及直接脂肪分数测量。

### 标签与定量单位

背景=0、IMAT=1、皮下脂肪=2、肌肉=3、骨=4。FF 百分数除以 100，体积单位为 mL。
`SMFF_wf` 为肌肉内 Fat 信号总和除以 Water+Fat 信号总和。

### 分割训练

使用 Adam，初始学习率 0.001，多项式学习率衰减；每层两次卷积，关闭深监督，
使用最终 epoch 检查点生成下游预测。输入 patch 为 71×432×432，内部补零到
96×448×448，输出裁回输入尺寸。详细算法见 [方法说明](methods.md)。

## 测试与数据

[软件测试说明](validation.md) 提供单元测试和合成影像测试命令。
仓库包含源码、配置、合成示例、测试和文档；受试者影像、临床记录及训练权重不随代码分发。
