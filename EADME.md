# NCD_NS_ALL7（改规则）

一个基于神经符号学习的认知诊断模型实现，结合逻辑规则约束进行学生知识掌握度评估。

## 项目结构

```
NCD_NS_ALL7（改规则）/
├── main.py                 # 训练与评估主程序
├── models.py              # NeuroSymbolicCD 模型实现
├── data_utils.py          # 数据加载与预处理工具
├── divide_data.py         # 数据集划分脚本
├── data/                  # 数据目录
│   ├── Assist/           # ASSISTments 数据集
│   ├── Math/             # Math 数据集
│   └── Junyi/            # Junyi 数据集
└── result/               # 模型结果保存目录
    ├── best_model.pth
    └── full_model_and_results.pth
```

## 功能特点

- **神经符号学习**：融合深度学习与逻辑规则
- **多种规则约束**：
  - 先决条件规则（Prerequisite）
  - 相似性配对（Similarity）
  - 组成规则（Compositional）
  - 平滑性与单调性约束
- **多数据集支持**：兼容 ASSISTments、Math 等数据集
- **早停机制**：防止过拟合
- **完整评估指标**：Accuracy、AUC、RMSE

## 快速开始

### 1. 数据准备

修改 `data_dir` 为你的数据目录（如 `'data/Assist'`）

### 2. 训练模型

```bash
python main.py
```

### 3. 关键参数配置

编辑 main.py 中的超参数：

```python
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
LAMBDA_LOGIC = 0.05      # 逻辑规则损失权重
PATIENCE = 3             # 早停忍耐轮数
```

## 数据格式

### 训练集 (train_set.json)

```json
[
  {
    "user_id": 1,
    "exer_id": 7,
    "score": 1,
    "knowledge_code": [1, 2]
  }
]
```

### 验证/测试集 (val_set.json / test_set.json)

```json
[
  {
    "user_id": 1,
    "log_num": 4,
    "logs": [
      {
        "exer_id": 7,
        "score": 1,
        "knowledge_code": [1, 2]
      }
    ]
  }
]
```

## 输出结果

训练完成后，结果保存在 `result/` 目录：

- `best_model.pth`：最佳模型权重
- `full_model_and_results.pth`：完整模型 + 配置 + 评估结果

## 性能指标

模型在测试集上输出：

- **Accuracy**：二分类准确率
- **AUC**：ROC 曲线下面积
- **RMSE**：均方根误差