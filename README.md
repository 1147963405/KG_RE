# KG_RE — 关系抽取工具

中文知识图谱关系抽取模块。基于级联二元标注框架，从中文临床文本中提取（主语, 关系, 宾语）三元组。

---

## 目录

- [模型下载](#模型下载)
- [依赖库](#依赖库)
- [项目结构](#项目结构)
- [模型使用](#模型使用)
  - [关系抽取](#关系抽取)
  - [训练](#训练)
  - [推理演示](#推理演示)
- [架构说明](#架构说明)
- [评估](#评估)

---

## 模型下载

预训练 BERT 权重及训练好的模型文件较大，已放置于百度网盘：

- **RE 模型**（含 `model_re.pkl` + `pytorch_model.bin`）：链接:https://pan.baidu.com/s/1cIse6JO2H78heXu7DNewmg 密码:4s6k

下载后请将文件放入 `model/medical_re/` 目录：

```
model/medical_re/
├── pytorch_model.bin     # BERT-base-chinese 预训练权重
├── model_re.pkl          # 训练好的级联模型检查点
├── config.json           # BERT 配置
├── vocab.txt             # BERT 词表
└── predicate.json        # 23 个关系谓词
```

## 依赖库

- Python >= 3.8
- torch
- transformers
- numpy
- json
- re
- random
- time
- gc

```bash
pip install torch transformers numpy
```

## 项目结构

```
KG_RE/
├── model/
│   └── medical_re/           # 预训练模型及配置文件
│       ├── config.json
│       ├── vocab.txt
│       ├── predicate.json
│       ├── model_re.pkl      # 训练好的模型（需下载）
│       ├── pytorch_model.bin # BERT 权重（需下载）
│       └── train_example.json # 训练数据示例（442 条）
├── model_re/                 # 源代码
│   ├── medical_re.py         # 核心模块：模型、数据管道、训练、推理
│   ├── train.py              # 训练入口
│   ├── demo_re.py            # 推理演示（5 个场景）
│   └── demo.py               # 简易推理演示
├── save/
│   └── model_re_trained.pkl  # 训练后输出的模型（占位）
├── docs/
│   └── clinical_relation_plan.md  # 谓词扩展方案（23→59）
└── utils/
    └── utils.py               # 分词评估工具（独立于管线）
```

## 模型使用

### 关系抽取

**配置路径**

在 `model_re/medical_re.py` 的 `class config` 中修改各文件路径为项目实际位置：

```python
PATH_SCHEMA = "D:/AI/KG_RE/model/medical_re/predicate.json"
PATH_TRAIN  = "D:/AI/KG_RE/model/medical_re/train_example.json"
PATH_BERT   = "D:/AI/KG_RE/model/medical_re/"
PATH_MODEL  = "D:/AI/KG_RE/model/medical_re/model_re.pkl"
PATH_SAVE   = "D:/AI/KG_RE/save/model_re_trained.pkl"
```

**推理**

```python
from model_re import medical_re

medical_re.load_schema()
model4s, model4po = medical_re.load_model()

text = "据报道称，新冠肺炎患者经常会发热、咳嗽，少部分患者会胸闷、乏力，其病因包括: 1.自身免疫系统缺陷\n2.人传人。"
res = medical_re.get_triples(text, model4s, model4po)
print(json.dumps(res, ensure_ascii=False, indent=2))
```

输出示例：

```json
[
  {
    "text": "据报道称，新冠肺炎患者经常会发热、咳嗽，少部分患者会胸闷、乏力，其病因包括: 1.自身免疫系统缺陷\n2.人传人",
    "triples": [
      ["新冠肺炎", "临床表现", "发热"],
      ["新冠肺炎", "临床表现", "咳嗽"],
      ["新冠肺炎", "临床表现", "胸闷"],
      ["新冠肺炎", "病因", "自身免疫系统缺陷"],
      ["新冠肺炎", "病因", "人传人"]
    ]
  }
]
```

**命令行运行**

```bash
# 推理演示（5 个场景：说明书文本、空输入、短句、长临床文档、JSON 格式化输出）
python model_re/demo_re.py

# 简易演示（单条 COVID-19 文本）
python model_re/demo.py
```

### 训练

```bash
python model_re/train.py
```

训练数据格式（`train_example.json`）：

```json
{
  "text": "阿莫西林常用于敏感菌所致的呼吸道感染...",
  "spo_list": [
    ["阿莫西林", "适应症", "呼吸道感染"],
    ["阿莫西林", "不良反应", "恶心"]
  ]
}
```

每个 epoch 输出 batch 级别损失，epoch 结束时打印总损失与耗时。训练完成后模型保存至 `PATH_SAVE`，并在 20% 验证集上评估 F1。

## 架构说明

```
输入文本
  │
  ▼ 按"。"分句，每句限长 128 字符
  │
  ▼
┌─────────────────────────────────────┐
│ extract_spoes(句子)                  │
│                                      │
│  tokenize → Model4s(BERT)           │
│    │                                  │
│    ▼ 主体预测（阈值 > 0.4）          │
│    │                                  │
│    ▼ 对每个主体：                     │
│    Model4p o(隐藏状态 + 主体位置)     │
│      │                                │
│      ▼ 客体+谓词预测（阈值 > 0.4）   │
│      │                                │
│      ▼ [(主体, 谓词, 客体), ...]     │
└─────────────────────────────────────┘

Model4s:  BERT → Dropout → Linear(768→2) → Sigmoid
Model4po: Hidden + SubjectEmb → Dropout → Linear(768→num_p×2) → Sigmoid
```

- 级联架构解耦了主体抽取与客体+关系预测，天然支持重叠关系
- 单句训练时随机选取一个主体作为监督；推理时遍历所有检测到的主体
- 当前模型在 CPU 上运行（所有 `.cuda()` 已注释），如需 GPU 请取消注释

## 评估

采用精确 SPO 三元组集合匹配：

```python
Precision = X / Y   # 预测正确数 / 总预测数
Recall    = X / Z   # 预测正确数 / 总标注数
F1        = 2 * X / (Y + Z)
```

在 `run_train()` 中自动以 8:2 划分训练/验证集，训练结束后打印评估结果。
